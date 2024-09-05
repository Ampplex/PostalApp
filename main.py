import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from web3 import Web3
import requests

# ML Model Code
data = {
    'weather_condition': [0, 1, 0, 1, 2, 0, 1, 2, 2, 0],
    'distance_km': [100, 200, 300, 400, 250, 120, 310, 450, 220, 150],
    'past_delays_hours': [1, 2, 0.5, 3, 2.5, 1, 0.5, 3.5, 2, 1.5],
    'route_blockages': [0, 1, 0, 1, 0, 0, 1, 1, 0, 0],
    'natural_calamities': [0, 0, 1, 1, 0, 0, 0, 1, 0, 0],
    'transport_cost': [5000, 10000, 15000, 20000, 12500, 6000, 15500, 22500, 11500, 7000],
    'optimal_transport_mode': [0, 1, 0, 1, 2, 0, 1, 2, 2, 0]
}

df = pd.DataFrame(data)
X = df[['weather_condition', 'distance_km', 'past_delays_hours', 'route_blockages', 'natural_calamities']]
y = df[['optimal_transport_mode', 'transport_cost']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultiOutputRegressor(LinearRegression())
model.fit(X_train, y_train)

# Example new data for prediction
new_data = np.array([[0, 150, 1, 0, 0]])
prediction = model.predict(new_data)

optimal_mode = np.round(prediction[0][0])
predicted_cost = prediction[0][1]

modes = {0: 'Truck', 1: 'Flight', 2: 'Train'}
print(f"Optimal Transport Mode: {modes[optimal_mode]}")
print(f"Predicted Transport Cost: ₹{predicted_cost:.2f}")

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

def book_transport(optimal_mode, predicted_cost, payment_id):
    sender_address = '0xYourSenderAddress'
    receiver_address = '0xYourReceiverAddress'
    admin_address = '0xYourAdminAddress'
    agency_address = '0xTransportAgencyAddress'

    if optimal_mode == 0:
        # Book Truck
        tx_hash = create_booking("Truck", predicted_cost, agency_address, admin_address)
        print("Truck Booking Transaction Hash:", tx_hash)
    elif optimal_mode == 1:
        # Book Flight
        tx_hash = create_booking("Flight", predicted_cost, agency_address, admin_address)
        print("Flight Booking Transaction Hash:", tx_hash)
    elif optimal_mode == 2:
        # Book Train
        tx_hash = create_booking("Train", predicted_cost, agency_address, admin_address)
        print("Train Booking Transaction Hash:", tx_hash)
    else:
        print("Invalid transport mode")
        return

    # Make Payment
    payment_tx_hash = make_payment(booking_id, sender_address)
    print("Payment Transaction Hash:", payment_tx_hash)

def create_booking(transport_mode, amount, transport_agency, admin):
    tx = booking_contract.functions.createBooking(transport_mode, amount, transport_agency).buildTransaction({
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
payment_id = process_payment(predicted_cost)

if payment_id:
    book_transport(optimal_mode, predicted_cost, payment_id)
else:
    print("Payment failed, booking not completed")
