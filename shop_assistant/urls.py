from django.urls import path
from . import views

app_name = 'shop_assistant'

urlpatterns = [
    path('',                            views.home,                 name='home'),
    path('recommend/',                  views.recommend,            name='recommend'),
    path('product/<str:product_name>/', views.product_detail,       name='product_detail'),
    path('api/recommend/',              views.api_recommend,        name='api_recommend'),
    path('api/stats/',                  views.api_stats,            name='api_stats'),
    path('api/categories/',             views.api_categories,       name='api_categories'),
    path('api/chat/',                   views.api_chat,             name='api_chat'),
    path('api/predict/',                views.api_predict_ethical,  name='api_predict_ethical'),
    path('about/',                      views.about,                name='about'),
]