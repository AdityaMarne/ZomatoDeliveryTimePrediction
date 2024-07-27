from flask import Flask,request,render_template,jsonify
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application


@app.route('/',methods=['GET','POST'])

def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')
    
    else:
        # Extracting data from the form
        data = CustomData(
            Delivery_person_Age=int(request.form.get('Delivery_person_Age')),
            Delivery_person_Ratings=float(request.form.get('Delivery_person_Ratings')),
            Restaurant_latitude=float(request.form.get('Restaurant_latitude')),
            Restaurant_longitude=float(request.form.get('Restaurant_longitude')),
            Delivery_location_latitude=float(request.form.get('Delivery_location_latitude')),
            Delivery_location_longitude=float(request.form.get('Delivery_location_longitude')),
            Weather_conditions=request.form.get('Weather_conditions'),
            multiple_deliveries=float(request.form.get('multiple_deliveries')),
            Festival=request.form.get('Festival'),
            City=request.form.get('City'),
            Road_traffic_density=request.form.get('Road_traffic_density'),
            Vehicle_condition=int(request.form.get('Vehicle_condition')),
            Type_of_vehicle=request.form.get('Type_of_vehicle')
        )
        
        # Converting data to dataframe for prediction
        final_new_data = data.get_data_as_dataframe()
        
        # Making prediction
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_new_data)
        
        # Rounding the result
        results = round(pred[0])
        
        # Rendering the result
        return render_template('form.html', final_result=f"This is the predicted time in min: {results} min")

    
if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)