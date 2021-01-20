from flask import Flask, render_template, request, send_from_directory
import pandas as pd
import os
import joblib


direc = os.getcwd()

prediction_model = joblib.load(direc + '\SMOTE_RandForClass_Model.pkl')
app = Flask(__name__)


@app.route('/')
@app.route('/home')
def homepage():
    return render_template('HomePage.html')

@app.route('/predict', methods =['GET', 'POST'])
def prediction():
    if request.method =='POST':
        if request.files:
            csvfile = request.files['csvfile']
            csvfile.save(os.path.join(direc, csvfile.filename))
            raw_data = pd.read_csv(csvfile.filename)
            columns = raw_data.columns.tolist()
            columns_needs = ['bidfloor', 'support_type','device_language', 'device_model', 'verticals_0', 'verticals_1', 'verticals_2', 'vertical_3', 'bid_price', 'won_price']
            x = 0
            for col in columns_needs:    
                for column in columns:
                    if col == column:
                        x+=1

            if x != 10:
                return render_template('Erreur.html')
            else:
                
                raw_data2 = raw_data[columns_needs]
                cat_vars=['device_model', 'support_type']
                for var in cat_vars:
                    cat_list='var'+'_'+var
                    cat_list = pd.get_dummies(raw_data[var], prefix=var)
                    data1=raw_data2.join(cat_list)
                    raw_data2=data1
                cat_vars=['device_model', 'support_type']
                data_vars=raw_data2.columns.values.tolist()
                to_keep=[i for i in data_vars if i not in cat_vars]
                final_data=raw_data2[to_keep]
                device_language = final_data['device_language'].values.tolist()
                for i in range(len(device_language)):
                    if device_language[i] == 'en_EN' or device_language[i] == 'fr_FR':
                        device_language[i] = 1 #big language
                    elif device_language[i] == 'ar_AR' or device_language[i] == 'es_ES' or device_language[i] == 'ru_RU' or device_language[i] == 'de_DE' or device_language[i] =='zh_CN' or device_language[i] == 'hi_HI':
                        device_language[i] = 2 #regional language
                    else:
                        device_language[i] = 3
        
                final_data = final_data.drop(['device_language'], axis = 1)
                final_data['dev_lang'] = device_language
                y_pred = pd.DataFrame(prediction_model.predict(final_data), columns = ['clicked'])
                raw_data['clicked'] = y_pred['clicked']
                raw_data.clicked = raw_data.clicked.astype(int)
                raw_data.to_csv('Result_'+ csvfile.filename, sep =',')
                return send_from_directory(direc,filename = 'Result_'+ csvfile.filename, as_attachment = True)
    return render_template("PredictionPage.html")


if __name__ == "__main__":
    app.run()
