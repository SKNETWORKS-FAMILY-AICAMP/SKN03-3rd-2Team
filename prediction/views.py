from django.shortcuts import render

# Create your views here.

def prediction(request): 
    return render(request, "prediction.html")