from flask import Flask, render_template, request
import pickle
import pandas as pd
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENCAGE_KEY")

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

app = Flask(__name__)


def render_page(prediction_text=None, log_values=None, status="idle"):
    return render_template(
        'index.html',
        form_values={},
        log_values=log_values or {},
        prediction_text=prediction_text,
        status=status
    )


#  FUNCTION: Get coordinates using OpenCage
def get_coordinates(place):
    if not API_KEY:
        return None

    url = "https://api.opencagedata.com/geocode/v1/json"
    if "," not in place:
        search_terms = [
            f"{place}, Pune, Maharashtra, India",
            f"{place}, Maharashtra, India",
            f"{place}, India",
            place
        ]
    else:
        search_terms = [place]

    for search_term in search_terms:
        params = {
            'q': search_term,
            'key': API_KEY,
            'countrycode': 'in',
            'limit': 1,
            'no_annotations': 1
        }

        try:
            response = requests.get(url, params=params, timeout=10)
        except requests.RequestException:
            continue

        if response.status_code != 200:
            continue

        data = response.json()

        if 'results' not in data or len(data['results']) == 0:
            continue

        lat = data['results'][0]['geometry']['lat']
        lng = data['results'][0]['geometry']['lng']

        return lat, lng

    return None


@app.route('/')
def home():
    return render_page()


@app.route('/predict', methods=['POST'])
def predict():
    print(request.form)
    try:
        form_data = request.form
        form_values = form_data.to_dict()

        # 🔹 Get numeric inputs
        age = float(form_data['f1'])
        rating = float(form_data['f2'])
        vehicle_condition = float(form_data['f7'])
        multiple_deliveries = float(form_data['f8'])

        # 🔹 Get location names
        restaurant = form_data['restaurant']
        delivery = form_data['delivery']

        # 🔹 Convert locations → coordinates
        rest_coords = get_coordinates(restaurant)
        del_coords = get_coordinates(delivery)

        if rest_coords is None or del_coords is None:
            return render_page(
                prediction_text="Location not found. Try adding city/state to the location names.",
                log_values=form_values,
                status="error"
            )

        rest_lat, rest_long = rest_coords
        del_lat, del_long = del_coords

        # 🔹 Feature dictionary (29 features EXACT)
        feature_dict = {
            'Delivery_person_Age': age,
            'Delivery_person_Ratings': rating,
            'Restaurant_latitude': rest_lat,
            'Restaurant_longitude': rest_long,
            'Delivery_location_latitude': del_lat,
            'Delivery_location_longitude': del_long,
            'Vehicle_condition': vehicle_condition,
            'multiple_deliveries': multiple_deliveries,

            'Weatherconditions_conditions Fog': 0,
            'Weatherconditions_conditions NaN': 0,
            'Weatherconditions_conditions Sandstorms': 0,
            'Weatherconditions_conditions Stormy': 0,
            'Weatherconditions_conditions Sunny': 0,
            'Weatherconditions_conditions Windy': 0,

            'Road_traffic_density_Jam ': 0,
            'Road_traffic_density_Low ': 0,
            'Road_traffic_density_Medium ': 1,
            'Road_traffic_density_NaN ': 0,

            'Type_of_order_Drinks ': 0,
            'Type_of_order_Meal ': 1,
            'Type_of_order_Snack ': 0,

            'Type_of_vehicle_electric_scooter ': 0,
            'Type_of_vehicle_motorcycle ': 1,
            'Type_of_vehicle_scooter ': 0,

            'Festival_No ': 1,
            'Festival_Yes ': 0,

            'City_NaN ': 0,
            'City_Semi-Urban ': 0,
            'City_Urban ': 1
        }

        # 🔹 Convert to DataFrame
        final_df = pd.DataFrame([feature_dict])

        # 🔹 Prediction
        prediction = model.predict(final_df)

        return render_page(
            prediction_text=f"Estimated Delivery Time: {prediction[0]:.2f} minutes",
            log_values=form_values,
            status="success"
        )

    except Exception as e:
        return render_page(
            prediction_text=f"Error: {str(e)}",
            log_values=request.form.to_dict(),
            status="error"
        )


if __name__ == "__main__":
    app.run(debug=True)
