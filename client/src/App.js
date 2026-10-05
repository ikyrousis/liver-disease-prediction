import React, { useState } from 'react';
import axios from 'axios';

const PatientForm = () => {
  const [gender, setGender] = useState('');
  const [age, setAge] = useState('');
  const [totalBilirubin, setTotalBilirubin] = useState('');
  const [directBilirubin, setDirectBilirubin] = useState('');
  const [alkalinePhosphotase, setAlkalinePhosphotase] = useState('');
  const [alamineAminotransferase, setAlamineAminotransferase] = useState('');
  const [aspartateAminotransferase, setAspartateAminotransferase] = useState('');
  const [totalProteins, setTotalProteins] = useState('');
  const [albumin, setAlbumin] = useState('');
  const [agRatio, setAgRatio] = useState('');

  const handleSubmit = async (event) => {
    event.preventDefault();
    try {
      const response = await axios.post('http://localhost:5000/liver-disease-prediction', {
        gender,
        age,
        'total_bilirubin': totalBilirubin,
        'direct_bilirubin': directBilirubin,
        'alkaline_phosphotase': alkalinePhosphotase,
        'alamine_aminotransferase': alamineAminotransferase,
        'aspartate_aminotransferase': aspartateAminotransferase,
        'total_proteins': totalProteins,
        'albumin': albumin,
        'albumin_and_globulin_ratio': agRatio,
      });

      // response.data now holds the prediction from the server
      alert(`The prediction is: ${response.data.prediction}`);
    } catch (error) {
      alert(`There was an error: ${error}`);
    }
  };

  return (
    <form onSubmit={handleSubmit} style={style}>
      <input style={inputStyle} placeholder="Gender" value={gender} onChange={e => setGender(e.target.value)} />
      <input style={inputStyle} placeholder="Age" value={age} onChange={e => setAge(e.target.value)} type="number" />
      <input style={inputStyle} placeholder="Total Bilirubin" value={totalBilirubin} onChange={e => setTotalBilirubin(e.target.value)} type="number" />
      <input style={inputStyle} placeholder="Direct Bilirubin" value={directBilirubin} onChange={e => setDirectBilirubin(e.target.value)} type="number" />
      <input style={inputStyle} placeholder="Alkaline Phosphotase" value={alkalinePhosphotase} onChange={e => setAlkalinePhosphotase(e.target.value)} type="number" />
      <input style={inputStyle} placeholder="Alamine Aminotransferase" value={alamineAminotransferase} onChange={e => setAlamineAminotransferase(e.target.value)} type="number" />
      <input style={inputStyle} placeholder="Aspartate Aminotransferase" value={aspartateAminotransferase} onChange={e => setAspartateAminotransferase(e.target.value)} type="number" />
      <input style={inputStyle} placeholder="Total Proteins" value={totalProteins} onChange={e => setTotalProteins(e.target.value)} type="number" />
      <input style={inputStyle} placeholder="Albumin" value={albumin} onChange={e => setAlbumin(e.target.value)} type="number" />
      <input style={inputStyle} placeholder="Albumin and Globulin Ratio" value={agRatio} onChange={e => setAgRatio(e.target.value)} type="number" />
      <button style={buttonStyle} type="submit">Submit</button>
    </form>
  );
};

export default PatientForm;


// Add some style
const style = {
  display: 'flex',
  flexDirection: 'column',
  maxWidth: '300px',
  margin: '0 auto',
  padding: '20px',
  borderRadius: '10px',
  backgroundColor: '#f0f0f0',
};

const inputStyle = {
  margin: '5px 0',
  padding: '10px',
  fontSize: '16px',
};

const buttonStyle = {
  marginTop: '10px',
  padding: '10px',
  backgroundColor: '#007bff',
  color: '#fff',
  fontSize: '16px',
  border: 'none',
  borderRadius: '5px',
  cursor: 'pointer',
};