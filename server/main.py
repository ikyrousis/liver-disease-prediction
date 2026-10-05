"""AI Liver Disease Prediction - Flask backend (Problog inference).

The server builds a Problog model from the Indian Liver Patient dataset and
serves predictions over POST /liver-disease-prediction.

Model construction:
- Patient lab values are binarized into boolean evidence terms (male, young,
  normalBilirubin, ...). A term holds when the value is at or below the median
  of the training split for the corresponding feature.
- One annotated-disjunction rule is produced per distinct evidence pattern:
      w::liver_disease ; (1-w)::healthy :- <evidence conjunction>.
  w is the empirical class posterior of that pattern in the training data.
- Evidence patterns absent from the training data are evaluated against the
  training-set class prior.
- Evidence literals use lowercase true/false, as Problog parses uppercase
  'True' as an unbound variable.
"""

import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from problog.program import PrologString
from problog import get_evaluatable
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from patient import Patient

app = Flask(__name__)
CORS(app)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILENAME = 'indian_liver_patient.csv'
MODEL_FILENAME = 'trained_model.pkl'
UNTRAINED_FILENAME = 'untrained_model.pl'
METRICS_FILENAME = 'metrics.txt'
LABEL_COLUMN = 'Dataset'
GENDER_COLUMN = 'Gender'

# Feature schema: (Problog evidence term, Patient attribute, CSV column).
# A term holds when the patient value is at or below the training-split median
# of the corresponding column.
FEATURE_TERMS = [
    ('young', 'age', 'Age'),
    ('normalBilirubin', 'total_bilirubin', 'Total_Bilirubin'),
    ('normalDirectBilirubin', 'direct_bilirubin', 'Direct_Bilirubin'),
    ('normalAlkalinePhosphotase', 'alkaline_phosphotase', 'Alkaline_Phosphotase'),
    ('normalAlamine', 'alamine_aminotransferase', 'Alamine_Aminotransferase'),
    ('normalAspartate', 'aspartate_aminotransferase', 'Aspartate_Aminotransferase'),
    ('normalTotalProteins', 'total_proteins', 'Total_Protiens'),
    ('normalAlbumin', 'albumin', 'Albumin'),
    ('normalAGRatio', 'albumin_and_globulin_ratio', 'Albumin_and_Globulin_Ratio'),
]
TERM_NAMES = ['male'] + [term for term, _, _ in FEATURE_TERMS]

# Model state (loaded at startup, or trained in the block below).
trained_program = None      # weighted Problog program (string)
thresholds = None           # Patient attribute -> median cutoff
known_signatures = None     # set of boolean evidence patterns seen in training
disease_prior = None        # P(liver patient) in the training split


def find_csv():
    candidates = [
        os.path.join(SCRIPT_DIR, CSV_FILENAME),
        os.path.join(os.getcwd(), CSV_FILENAME),
        os.path.join(os.path.dirname(SCRIPT_DIR), CSV_FILENAME),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f'{CSV_FILENAME} not found. Looked in: {", ".join(candidates)}')


def evidence_atoms(patient, cutoffs):
    """Boolean evidence atoms for one patient, in the fixed term order."""
    atoms = [('male', str(patient.gender).lower() == 'male')]
    for term, attr, _ in FEATURE_TERMS:
        atoms.append((term, float(getattr(patient, attr)) <= cutoffs[attr]))
    return atoms


def signature_body(signature):
    """Problog conjunction for a boolean signature, e.g. '\\+male, young, ...'."""
    return ','.join(term if positive else '\\+' + term
                    for term, positive in zip(TERM_NAMES, signature))


def patient_from_row(row):
    return Patient(row[GENDER_COLUMN], row['Age'],
                   *[row[column] for _, attr, column in FEATURE_TERMS if attr != 'age'])


def evaluate_patient(patient):
    """Return (P(liver_disease), P(healthy), matched_pattern)."""
    atoms = evidence_atoms(patient, thresholds)
    signature = tuple(positive for _, positive in atoms)
    if signature not in known_signatures:
        # Evidence pattern absent from the training data: the class prior
        # applies, and the pattern is reported as unmatched.
        return disease_prior, 1.0 - disease_prior, False
    program = trained_program + '\n'
    program += ''.join(f'evidence({term},{str(value).lower()}).\n' for term, value in atoms)
    program += 'query(liver_disease).\nquery(healthy).'
    result = get_evaluatable().create_from(PrologString(program), propagate_evidence=True).evaluate()
    probabilities = {str(atom): value for atom, value in result.items()}
    return (probabilities.get('liver_disease', 0.0),
            probabilities.get('healthy', 0.0),
            True)


def prediction_message(p_liver, p_healthy, matched):
    message = (f'Probability of liver disease: {p_liver:.4f}\n'
               f'Probability to be healthy: {p_healthy:.4f}\n')
    if not matched:
        message += '(note: evidence pattern not present in training data - prior used)\n'
    return message


def submitPatient(person):
    message = prediction_message(*evaluate_patient(person))
    print(message)
    return message


def getProbabilities(person):
    p_liver, p_healthy, _ = evaluate_patient(person)
    return 1 if p_liver > p_healthy else 2


model_path = os.path.join(SCRIPT_DIR, MODEL_FILENAME)

model_bundle = None
if os.path.exists(model_path):
    try:
        with open(model_path, 'rb') as f:
            model_bundle = pickle.load(f)
    except Exception:
        model_bundle = None

if isinstance(model_bundle, dict) and 'program' in model_bundle:
    print('Found the trained model! Load in the system!')
    trained_program = model_bundle['program']
    thresholds = model_bundle['thresholds']
    known_signatures = model_bundle['signatures']
    disease_prior = model_bundle['prior']
else:
    print('There is no trained model, training now!')

    csv_path = find_csv()
    data = pd.read_csv(csv_path).dropna()
    # Stratified 70/30 split. NOTE: passing stratify as a numpy array vs a pandas
    # Series yields different partitions on current scikit-learn; the array form
    # is pinned here so the split (and metrics) are reproducible.
    train_idx, test_idx = train_test_split(
        np.arange(len(data)), test_size=0.3, random_state=42,
        stratify=(data[LABEL_COLUMN] == 1).astype(int).values)
    train_data = data.iloc[train_idx].reset_index(drop=True)
    test_data = data.iloc[test_idx].reset_index(drop=True)

    # Binarization cutoffs: the median of each lab value in the training split.
    thresholds = {attr: float(train_data[column].median())
                  for _, attr, column in FEATURE_TERMS}
    disease_prior = float((train_data[LABEL_COLUMN] == 1).mean())

    # Count how often each evidence pattern occurs per class...
    counts = {}
    for _, row in train_data.iterrows():
        signature = tuple(positive for _, positive in
                          evidence_atoms(patient_from_row(row), thresholds))
        bucket = counts.setdefault(signature, [0, 0])  # [n_healthy, n_liver]
        bucket[1 if int(row[LABEL_COLUMN]) == 1 else 0] += 1
    known_signatures = set(counts)

    # ...and set each rule weight to the empirical class posterior of its pattern.
    weighted_rules = []
    for signature in sorted(counts):
        n_healthy, n_liver = counts[signature]
        p_liver = n_liver / (n_healthy + n_liver)
        weighted_rules.append(
            f'{p_liver:.6f}::liver_disease ; {1.0 - p_liver:.6f}::healthy :- '
            f'{signature_body(signature)}.')

    # The atom priors are inert at inference (evidence fixes every atom) and are
    # retained for structural completeness of the program.
    trained_program = '\n'.join(
        [f'{(train_data[GENDER_COLUMN].str.lower() == "male").mean():.6f}::male.'] +
        [f'{(train_data[column] <= thresholds[attr]).mean():.6f}::{term}.'
         for term, attr, column in FEATURE_TERMS] +
        weighted_rules)

    # Untrained model artifact: program structure with placeholder weights.
    untrained_rules = [f't(_)::{term}.' for term in TERM_NAMES] + \
        [f't(_)::liver_disease ; t(_)::healthy :- {signature_body(signature)}.'
         for signature in sorted(counts)]
    with open(os.path.join(SCRIPT_DIR, UNTRAINED_FILENAME), 'w') as f:
        f.write('\n'.join(untrained_rules) + '\n')
    print('Untrained Model created')

    # Persist the model together with everything needed to serve patients.
    model_bundle = {'program': trained_program, 'thresholds': thresholds,
                    'signatures': known_signatures, 'prior': disease_prior}
    with open(model_path, 'wb') as f:
        pickle.dump(model_bundle, f)
    print('Trained Model created')

    # Evaluate on the held-out test split and write the metrics artifacts.
    test_patients = [patient_from_row(row) for _, row in test_data.iterrows()]
    actual_classifications = [int(row[LABEL_COLUMN]) for _, row in test_data.iterrows()]
    predicted_classifications = [getProbabilities(patient) for patient in test_patients]

    acc = accuracy_score(actual_classifications, predicted_classifications)
    prec = precision_score(actual_classifications, predicted_classifications, average='macro')
    rec = recall_score(actual_classifications, predicted_classifications, average='macro')
    f1 = f1_score(actual_classifications, predicted_classifications, average='macro')
    cm = confusion_matrix(actual_classifications, predicted_classifications)

    with open(os.path.join(SCRIPT_DIR, METRICS_FILENAME), 'w') as f:
        f.write(f'Accuracy: {acc}\n')
        f.write(f'Precision: {prec}\n')
        f.write(f'Recall: {rec}\n')
        f.write(f'F1 Score: {f1}\n')
        f.write(f'Confusion Matrix: \n{cm}\n')
    print(f'Test accuracy: {acc:.4f} (metrics written to {METRICS_FILENAME})')

    metrics = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1 Score': f1}
    plt.figure(figsize=(10, 5))
    plt.bar(metrics.keys(), metrics.values(), color=['blue', 'purple', 'orange', 'green'])
    plt.title('Model Performance Metrics')
    plt.ylabel('Score')
    plt.xlabel('Metrics')
    plt.ylim([0, 1])

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


@app.route('/liver-disease-prediction', methods=['POST'])
def predict_liver_disease():
    jsonData = request.get_json(force=True)  # forcing the interpretation of request data as JSON

    # Create a patient object with the provided data
    patient = Patient(
        jsonData['gender'],
        int(jsonData['age']),
        float(jsonData['total_bilirubin']),
        float(jsonData['direct_bilirubin']),
        float(jsonData['alkaline_phosphotase']),
        float(jsonData['alamine_aminotransferase']),
        float(jsonData['aspartate_aminotransferase']),
        float(jsonData['total_proteins']),
        float(jsonData['albumin']),
        float(jsonData['albumin_and_globulin_ratio'])
    )

    # Evaluate the patient and return the prediction response.
    p_liver, p_healthy, matched = evaluate_patient(patient)
    prediction = prediction_message(p_liver, p_healthy, matched)
    return jsonify({
        'prediction': prediction,
        'probability_liver_disease': p_liver,
        'probability_healthy': p_healthy,
        'matched_training_pattern': matched
    })


if __name__ == '__main__':
    # Default configuration: port 5000, loopback interface, debug enabled.
    # HOST, PORT and FLASK_DEBUG may be overridden (e.g. by Docker Compose).
    app.run(
        port=int(os.environ.get('PORT', 5000)),
        host=os.environ.get('HOST', '127.0.0.1'),
        debug=os.environ.get('FLASK_DEBUG', 'true').lower() in ('1', 'true', 'yes'),
    )
