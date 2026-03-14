from django.urls import path
from . import views

app_name = 'shop_assistant'

urlpatterns = [
    path('',                    views.home,                 name='home'),
    path('recommend/',          views.recommend,            name='recommend'),
    path('product/<str:product_name>/', views.product_detail, name='product_detail'),
    path('api/recommend/',      views.api_recommend,        name='api_recommend'),
    path('api/stats/',          views.api_stats,            name='api_stats'),
    path('api/categories/',     views.api_categories,       name='api_categories'),
    path('api/chat/',           views.api_chat,             name='api_chat'),
    path('api/predict/',        views.api_predict_ethical,  name='api_predict_ethical'),
    path('about/',              views.about,                name='about'),
    path('scan/',               views.scan,                 name='scan'),
    path('api/scan/',           views.api_scan,             name='api_scan'),
    path('signin/',             views.login_page,           name='signin'),
    path('api/auth/send-otp/',  views.api_send_otp,         name='api_send_otp'),
    path('api/auth/verify-otp/',views.api_verify_otp,       name='api_verify_otp'),
    path('api/auth/logout/',    views.api_logout,            name='api_logout'),
    path('api/auth/status/',    views.api_auth_status,       name='api_auth_status'),
    path('api/save-product/',   views.api_save_product,      name='api_save_product'),
    path('api/my-products/',    views.api_my_products,       name='api_my_products'),
    path('api/barcode/',         views.api_barcode_lookup,    name='api_barcode_lookup'),
]
