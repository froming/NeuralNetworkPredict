from django.shortcuts import render

def page_not_found(request,expresion):
    return render(request,'page_not_found.html')