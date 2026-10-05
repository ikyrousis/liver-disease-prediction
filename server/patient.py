class Patient:
    def __init__(self, gender, age, total_bilirubin, direct_bilirubin,
                 alkaline_phosphotase, alamine_aminotransferase, aspartate_aminotransferase, total_proteins,
                 albumin, albumin_and_globulin_ratio):
        self.gender = gender
        self.age = int(age) if age is not None else None
        self.total_bilirubin = total_bilirubin
        self.direct_bilirubin = direct_bilirubin
        self.alkaline_phosphotase = alkaline_phosphotase
        self.alamine_aminotransferase = alamine_aminotransferase
        self.aspartate_aminotransferase = aspartate_aminotransferase
        self.total_proteins = total_proteins
        self.albumin = albumin
        self.albumin_and_globulin_ratio = albumin_and_globulin_ratio
