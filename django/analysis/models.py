from django.db import models

class CustomerChurn(models.Model):
    
    customerID = models.CharField(max_length=20, primary_key=True)
    gender = models.CharField(max_length=10)
    SeniorCitizen = models.BooleanField(default=False)
    Partner = models.BooleanField(default=False)
    Dependents = models.BooleanField(default=False)
    tenure = models.IntegerField()
    PhoneService = models.CharField(max_length=20)
    MultipleLines = models.CharField(max_length=20)
    InternetService = models.CharField(max_length=20)
    OnlineSecurity = models.CharField(max_length=20)
    OnlineBackup = models.CharField(max_length=20)
    DeviceProtection = models.CharField(max_length=20)
    TechSupport = models.CharField(max_length=20)
    StreamingTV = models.CharField(max_length=20)
    StreamingMovies = models.CharField(max_length=20)
    Contract = models.CharField(max_length=20)
    PaperlessBilling = models.CharField(max_length=10)
    PaymentMethod = models.CharField(max_length=20)
    MonthlyCharges = models.FloatField()
    TotalCharges = models.FloatField()
    Churn = models.CharField(max_length=3) 
    class Meta:
        db_table = 'teleco'

    def __str__(self):
        return self.customer_id
