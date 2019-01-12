from rest_framework.generics import ListAPIView, RetrieveAPIView
from django.http import HttpResponse
from ..models import Article
from .serializers import ArticleSerializer
import json
import pandas as pd  # libreria que permite manipular archivos
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

class ArticleListView(ListAPIView):
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer

class ArticleDetailsView(RetrieveAPIView):
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer

def training_model(array_to_train):
    df = pd.read_csv("./mastitis.csv")  # cargar los datos sobre mastitis
    df = df.set_index('ID_muestra')
    feature_col_names = ['ED',
                         'DEL',
                         'NP',
                         'PL',
                         'CE',
                         'CCS',
                         'SCCS']
    predicted_class_name = ['Resultado']
    X = df[feature_col_names].values
    y = df[predicted_class_name].values
    split_test_size = 0.30
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_test_size, random_state=45)
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train.ravel())
    nb_predict_test = nb_model.predict(X_test)
    confusion_matrix = (metrics.confusion_matrix(y_test, nb_predict_test))
    classification_report = (metrics.accuracy_score(y_test, nb_predict_test))
    nb_predict_test_entered = nb_model.predict([array_to_train])  
    prediction_report = [
        {"test_result": nb_predict_test_entered},
        {"class_report": classification_report},
        {"confusion_matrix": confusion_matrix}
    ]
    return prediction_report


def model_predict(request):
    request_data = json.loads(request.body.decode('utf-8'))['values']
    if request.method == "POST":
        array_values = [
            float(request_data['ed']),
            float(request_data['del']),
            float(request_data['np']),
            float(request_data['pl']),
            float(request_data['ce']),
            float(request_data['ccs']),
            float(request_data['sccs']),
        ]
    response_values = training_model(array_values)
    true_positives = response_values[2]['confusion_matrix'][0][0]
    false_positives = response_values[2]['confusion_matrix'][0][1]
    false_negatives = response_values[2]['confusion_matrix'][1][0]
    true_negatives = response_values[2]['confusion_matrix'][1][1]

    result_decease = response_values[0]['test_result'][0]
    class_report = response_values[1]['class_report']
    response_object_json = {
        "TruePositives:"+ str(true_positives)+',',
        "TrueNegatives:"+str(true_negatives)+',',
        "FalseNegatives:"+str(false_negatives)+',',
        "FalsePositives:"+str(false_positives)+',',
        "ClassReport:"+str(class_report)+',',
        "ResultDeseace:"+str(result_decease)+','}
    return HttpResponse(response_object_json)
