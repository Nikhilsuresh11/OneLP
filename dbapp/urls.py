from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from .views import summarize_url,home,contact,voice,summarize_text,sentiment,summarize_document,create_blog,sentiment_analysis,senti_home,chat_view

urlpatterns = [
    path('', home, name='home'),
    path('summarize/', summarize_url, name='summarize'),
    path('summarize_text/', summarize_text, name='summarize_text'),
    path('summarize_document/', summarize_document, name='summarize_document'),
    path('sentiment/', sentiment, name='sentiment'),
    path('senti_home/', senti_home, name='senti_home'),
    path('create_blog/', create_blog, name='create_blog'),
    path('contact/', contact, name='contact'),
    path('voice/', voice, name='voice'),
    path('sentiment_analysis/', sentiment_analysis, name='sentiment_analysis'),
    path('chat/', chat_view, name='chat'),



]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
