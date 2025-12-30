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







class RegressionForm(forms.Form):
    work_year = forms.IntegerField(label="Year of Work", min_value=2000, max_value=2030)
    
    EXPERIENCE_CHOICES = [
        ('EN', 'Entry-level'),
        ('MI', 'Mid-level'),
        ('SE', 'Senior-level'),
        ('EX', 'Executive'),
    ]
    experience_level = forms.ChoiceField(choices=EXPERIENCE_CHOICES, label="Experience Level")
    
    EMPLOYMENT_CHOICES = [
        ('FT', 'Full-time'),
        ('PT', 'Part-time'),
        ('CT', 'Contract'),
        ('FL', 'Freelance'),
    ]
    employment_type = forms.ChoiceField(choices=EMPLOYMENT_CHOICES, label="Employment Type")
    
    remote_ratio = forms.IntegerField(label="Remote Ratio (%)", min_value=0, max_value=100)
    
    COMPANY_SIZE_CHOICES = [
        ('S', 'Small'),
        ('M', 'Medium'),
        ('L', 'Large'),
    ]
    company_size = forms.ChoiceField(choices=COMPANY_SIZE_CHOICES, label="Company Size")
    
    employee_residence = forms.CharField(label="Employee Residence", max_length=50)
    company_location = forms.CharField(label="Company Location", max_length=50)
    job_title = forms.CharField(label="Job Title", max_length=100)
    salary_currency = forms.CharField(label="Salary Currency", max_length=10)


class KMeansForm(forms.Form):
    skill = forms.FloatField()
    salary = forms.FloatField()
