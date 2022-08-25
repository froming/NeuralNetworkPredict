from email.policy import default
from django.shortcuts import render

from django.http import HttpResponse

import re
import os

# Create your views here.
def download(request):
    try:
        if str(request.session['IsLogined']) == str(request.COOKIES['Form1ng']):
            if request.method == 'GET':
                filename = request.GET.get('filename',default='static/PredictImage/E_n_m_dataset.csv')
                if filename == 'static/PredictImage/E_n_m_dataset.csv':
                    with open(filename,'rb') as Form1ng:
                        response = HttpResponse(Form1ng.read())
                    response['Content-Type'] = 'application/csv'
                    response['Content-Disposition'] = "attachment;filename=" + 'E_n_m_dataset.csv'
                else:
                    with open('static/PredictImage/' + filename,'rb') as Form1ng:
                        response = HttpResponse(Form1ng.read())
                    response['Content-Type'] = 'image/png'
                    response["Content-Disposition"] = "attachment;filename=" + 'result.png'
                return response
            elif str(request.POST['ptfilename']).split('.')[0] != 'maml5Layer' and str(request.POST['ptfilename']).split('.')[1] == 'pt' and os.path.exists('train/upload/' + request.POST['ptfilename']) and len(str(request.POST['ptfilename']).split('.')) == 2 and not re.search(';|&|\|',str(request.POST['modelname']).split('.')[0]):
                filename = str(request.POST['ptfilename']).split('.')[0]
                filepath = 'static/PredictImage/' + filename + '.txt'
                with open(filepath, 'rb') as Form1ng:
                    response = HttpResponse(Form1ng.read())
                response['Content-Type'] = 'text/plain'
                response['Content-Disposition'] = 'attachment;filename=' + 'result.txt'
        else:
            return render(request, 'login.html')
    except:
        return render(request, 'login.html')