from django.urls import path
from .views import *

app_name = 'main'

urlpatterns= [
    path('', show_main, name='show_main'),
]