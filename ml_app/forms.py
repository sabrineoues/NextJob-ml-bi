from django import forms


class KNNSampleForm(forms.Form):
    company_name = forms.CharField(label="Company Name")
    location = forms.CharField(label="Location")
    salary = forms.FloatField(label="Salary")





class XGBForm(forms.Form):
    location = forms.CharField(label="Location")
    skill = forms.CharField(label="Skill")
    company_name = forms.CharField(label="Company Name")
    platform_name = forms.CharField(label="Platform Name")
    degree = forms.CharField(label="Degree")
    salary = forms.FloatField(label="Salary")  # si tu veux garder le salaire




class RegressionForm(forms.Form):
    EXPERIENCE_LEVEL_CHOICES = [
        ('EN', 'Entry'),
        ('MI', 'Mid'),
        ('SE', 'Senior'),
        ('EX', 'Executive'),
    ]
    COMPANY_SIZE_CHOICES = [
        ('S', 'Small'),
        ('M', 'Medium'),
        ('L', 'Large'),
    ]
    EMPLOYMENT_TYPE_CHOICES = [
        ('FT', 'Full-Time'),
        ('PT', 'Part-Time'),
        ('CT', 'Contract'),
        ('FL', 'Freelance'),
    ]
    
    experience_level = forms.ChoiceField(choices=EXPERIENCE_LEVEL_CHOICES, label="Experience Level")
    company_size = forms.ChoiceField(choices=COMPANY_SIZE_CHOICES, label="Company Size")
    employment_type = forms.ChoiceField(choices=EMPLOYMENT_TYPE_CHOICES, label="Employment Type")
    work_year = forms.IntegerField(label="Work Year")
    employee_residence = forms.CharField(label="Employee Residence")
    company_location = forms.CharField(label="Company Location")
    remote_ratio = forms.IntegerField(label="Remote Ratio (0-100)")
    job_title = forms.CharField(label="Job Title")
    salary_currency = forms.CharField(label="Salary Currency")



class KMeansForm(forms.Form):
    skill = forms.FloatField()
    salary = forms.FloatField()
