import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
import requests
from web3 import Web3

# Updated Data with Duration and Weather Impact
data = {
    'weather_condition': [0, 1, 0, 1, 2, 0, 1, 2, 2, 0],
    'distance_km': [100, 200, 300, 400, 250, 120, 310, 450, 220, 150],
    'past_delays_hours': [1, 2, 0.5, 3, 2.5, 1, 0.5, 3.5, 2, 1.5],
    'route_blockages': [0, 1, 0, 1, 0, 0, 1, 1, 0, 0],
    'natural_calamities': [0, 0, 1, 1, 0, 0, 0, 1, 0, 0],
    'duration_hours': [2, 4, 5, 7, 5, 3, 6, 8, 4, 3],
    'transport_cost': [5000, 10000, 15000, 20000, 12500, 6000, 15500, 22500, 11500, 7000],
    'optimal_transport_mode': [0, 1, 0, 1, 2, 0, 1, 2, 2, 0]
}

df = pd.DataFrame(data)
X = df[['weather_condition', 'distance_km', 'past_delays_hours', 'route_blockages', 'natural_calamities', 'duration_hours']]
y = df[['optimal_transport_mode', 'transport_cost']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultiOutputRegressor(LinearRegression())
model.fit(X_train, y_train)

# Example new data for prediction
new_data = np.array([[0, 150, 1, 0, 0, 3]])  # Added Duration and Weather
prediction = model.predict(new_data)

optimal_mode = np.round(prediction[0][0])
predicted_cost = prediction[0][1]

# Considering weather impact
weather_impact = {0: 0.9, 1: 1.0, 2: 1.2}  # Weather impact on cost and duration
weather_condition = new_data[0][0]
cost_with_weather = predicted_cost * weather_impact[weather_condition]
duration_with_weather = new_data[0][5] * weather_impact[weather_condition]

modes = {0: 'Truck', 1: 'Flight', 2: 'Train'}
print(f"Optimal Transport Mode: {modes[optimal_mode]}")
print(f"Predicted Transport Cost (with Weather Impact): ₹{cost_with_weather:.2f}")
print(f"Predicted Duration (with Weather Impact): {duration_with_weather:.2f} hours")

# API URLs for truck, flight, and train booking
truck_api_url = "https://api.trukky.com/v1/book"
flight_api_url = "https://partners.api.skyscanner.net/apiservices/pricing/v1.0"
train_api_url = "https://api.irctc.co.in/book"

# Blockchain Integration
web3 = Web3(Web3.HTTPProvider('YOUR_INFURA_OR_ALCHEMY_URL'))

parcel_abi = [...]  # ABI for ParcelTracking contract
parcel_address = 'YOUR_CONTRACT_ADDRESS'
parcel_contract = web3.eth.contract(address=parcel_address, abi=parcel_abi)

booking_abi = [...]  # ABI for TransportBooking contract
booking_address = 'YOUR_CONTRACT_ADDRESS'
booking_contract = web3.eth.contract(address=booking_address, abi=booking_abi)

def process_payment(amount):
    api_key = "YOUR_PAYMENT_GATEWAY_API_KEY"
    url = "https://api.paymentgateway.com/v1/payments"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payment_data = {
        "amount": amount,
        "currency": "INR",
        "description": "Transport booking payment"
    }
    response = requests.post(url, json=payment_data, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        payment_id = data['id']
        print("Payment processed successfully with ID:", payment_id)
        return payment_id
    else:
        print("Payment processing failed")
        return None

def book_transport(optimal_mode, cost_with_weather, payment_id):
    sender_address = '0xYourSenderAddress'
    admin_address = '0xYourAdminAddress'
    
    # Priority based on predicted cost and duration with weather impact
    transport_modes = {
        0: "Truck",
        1: "Flight",
        2: "Train"
    }
    
    if optimal_mode == 0:
        api_url = truck_api_url
    elif optimal_mode == 1:
        api_url = flight_api_url
    else:
        api_url = train_api_url
    
    booking_data = {
        'mode': transport_modes[optimal_mode],
        'amount': cost_with_weather,
        'admin': admin_address
    }
    
    response = requests.post(api_url, json=booking_data)
    
    if response.status_code == 200:
        tx_hash = response.json().get("transaction_hash")
        print(f"{transport_modes[optimal_mode]} Booking Transaction Hash:", tx_hash)
        
        # Blockchain booking creation
        tx = create_booking(transport_modes[optimal_mode], cost_with_weather, admin_address)
        print(f"Blockchain Transaction Hash: {tx}")
        
        # Make Payment
        payment_tx_hash = make_payment(payment_id, sender_address)
        print("Payment Transaction Hash:", payment_tx_hash)
    else:
        print(f"{transport_modes[optimal_mode]} booking failed")

def create_booking(transport_mode, amount, admin):
    tx = booking_contract.functions.createBooking(transport_mode, amount, admin).buildTransaction({
        'from': admin,
        'gas': 2000000,
        'gasPrice': web3.toWei('20', 'gwei'),
        'nonce': web3.eth.getTransactionCount(admin),
    })
    signed_tx = web3.eth.account.signTransaction(tx, private_key='YOUR_PRIVATE_KEY')
    tx_hash = web3.eth.sendRawTransaction(signed_tx.rawTransaction)
    return web3.toHex(tx_hash)

def make_payment(booking_id, payer):
    tx = booking_contract.functions.makePayment(booking_id).buildTransaction({
        'from': payer,
        'value': web3.toWei(predicted_cost, 'ether'),
        'gas': 2000000,
        'gasPrice': web3.toWei('20', 'gwei'),
        'nonce': web3.eth.getTransactionCount(payer),
    })
    signed_tx = web3.eth.account.signTransaction(tx, private_key='YOUR_PRIVATE_KEY')
    tx_hash = web3.eth.sendRawTransaction(signed_tx.rawTransaction)
    return web3.toHex(tx_hash)

# Execute the process
payment_id = process_payment(cost_with_weather)

if payment_id:
    book_transport(optimal_mode, cost_with_weather, payment_id)
else:
    print("Payment failed, booking not completed")
