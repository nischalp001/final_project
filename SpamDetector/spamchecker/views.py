from django.shortcuts import render
from django import forms
from .forms import EmailForm
import joblib

# Load the trained model and vectorizer
model = joblib.load('spamchecker/spam_model.pkl')
vectorizer = joblib.load('spamchecker/tfidf_vectorizer.pkl')

def check_spam(request):
    result = None
    if request.method == 'POST':
        form = EmailForm(request.POST)
        if form.is_valid():
            email_text = form.cleaned_data['email_text']
            email_vectorized = vectorizer.transform([email_text])
            prediction = model.predict(email_vectorized)
            result = 'Spam' if prediction[0] == 1 else 'Not Spam'
    else:
        form = EmailForm()
    
    return render(request, 'spamchecker/check_spam.html', {'form': form, 'result': result})
