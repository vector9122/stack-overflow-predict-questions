#from django.http import HttpResponse
from django.shortcuts import render
import joblib
import numpy
def home(request):
    return render(request,"index.html")

def result(request):
    predictor = joblib.load('Model.pkl')
    li = []
    li.append(request.GET['reputation'])
    li.append(request.GET['deleted_post'])
    li.append(len(request.GET['title']))
    new = numpy.array(li).reshape(1,3)
    ans = predictor.predict(new)[0]
    return render(request,"result.html", {'ans':ans})

