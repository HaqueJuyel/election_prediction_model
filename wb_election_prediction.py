# ==========================================================
# WEST BENGAL 294 SEAT ADVANCED ELECTION ML MODEL
# FULL VERSION WITH ALL SEAT OUTPUT
# ==========================================================

import pandas as pd
import numpy as np
import random

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


# ==========================================================
# WEST BENGAL CONSTITUENCIES (294)
# ==========================================================

constituencies = [

    "Alipurduars","Falakata","Madarihat","Kalchini","Kumargram",
    "Dhupguri","Maynaguri","Jalpaiguri","Rajganj","Dabgram-Phulbari",
    "Mal","Nagrakata","Kalimpong","Darjeeling","Kurseong",
    "Matigara-Naxalbari","Siliguri","Phansidewa","Chopra","Islampur",
    "Goalpokhar","Chakulia","Karandighi","Hemtabad","Kaliaganj",
    "Raiganj","Itahar","Kushmandi","Gangarampur","Harirampur",
    "Balurghat","Tapan","Kumarganj","Chanchal","Harishchandrapur",
    "Maldaha","Ratua","Manikchak","Maldaha Town","English Bazar",
    "Mothabari","Sujapur","Baisnabnagar","Farakka","Samserganj",
    "Suti","Jangipur","Raghunathganj","Sagardighi","Lalgola",
    "Bhagabangola","Raninagar","Murshidabad","Nabagram","Khargram",
    "Burwan","Kandi","Bharatpur","Rejinagar","Beldanga",
    "Baharampur","Hariharpara","Nowda","Domkal","Jalangi",
    "Karimpur","Tehatta","Palashipara","Kaliganj","Nakashipara",
    "Chapra","Krishnanagar Uttar","Nabadwip","Krishnanagar Dakshin",
    "Santipur","Ranaghat Uttar Paschim","Krishnaganj",
    "Ranaghat Uttar Purba","Ranaghat Dakshin","Chakdaha",
    "Kalyani","Haringhata","Bagdah","Bongaon Uttar",
    "Bongaon Dakshin","Gaighata","Swarupnagar","Baduria",
    "Habra","Ashoknagar","Amdanga","Bijpur","Naihati",
    "Bhatpara","Jagatdal","Noapara","Barrackpore","Khardaha",
    "Dum Dum Uttar","Panihati","Kamarhati","Baranagar",
    "Dum Dum","Rajarhat New Town","Bidhannagar",
    "Rajarhat Gopalpur","Madhyamgram","Barasat","Deganga",
    "Haroa","Minakhan","Sandeshkhali","Basirhat Dakshin",
    "Basirhat Uttar","Hingalganj","Gosaba","Basanti",
    "Kultali","Patharpratima","Kakdwip","Sagar","Kulpi",
    "Raidighi","Mandirbazar","Jaynagar","Baruipur Purba",
    "Canning Paschim","Canning Purba","Baruipur Paschim",
    "Magrahat Purba","Magrahat Paschim","Diamond Harbour",
    "Falta","Satgachia","Bishnupur","Maheshtala","Budge Budge",
    "Metiabruz","Kolkata Port","Bhabanipur","Rashbehari",
    "Ballygunge","Howrah Uttar","Howrah Madhya","Shibpur",
    "Howrah Dakshin","Sankrail","Panchla","Uluberia Purba",
    "Uluberia Uttar","Uluberia Dakshin","Shyampur","Bagnan",
    "Amta","Udaynarayanpur","Jagatballavpur","Domjur",
    "Uttarpara","Sreerampur","Champdani","Singur",
    "Chandannagar","Chunchura","Balagarh","Pandua",
    "Saptagram","Chanditala","Jangipara","Haripal",
    "Dhanekhali","Tarakeswar","Pursurah","Arambagh",
    "Goghat","Khanakul","Tamluk","Panskura Purba",
    "Panskura Paschim","Moyna","Nandakumar","Mahisadal",
    "Haldia","Nandigram","Chandipur","Patashpur",
    "Kanthi Uttar","Bhagabanpur","Khejuri",
    "Kanthi Dakshin","Ramnagar","Egra","Dantan",
    "Nayagram","Gopiballavpur","Jhargram","Keshiary",
    "Kharagpur Sadar","Narayangarh","Sabang","Pingla",
    "Kharagpur","Debra","Daspur","Ghatal",
    "Chandrakona","Garbeta","Salboni","Medinipur",
    "Binpur","Bandwan","Balarampur","Baghmundi",
    "Joypur","Purulia","Manbazar","Kashipur",
    "Para","Raghunathpur","Saltora","Chhatna",
    "Ranibandh","Raipur","Taldangra","Bankura",
    "Barjora","Onda","Bishnupur Bankura","Katulpur",
    "Indas","Sonamukhi","Khandaghosh",
    "Bardhaman Dakshin","Raina","Jamalpur",
    "Monteswar","Kalna","Memari","Bardhaman Uttar",
    "Bhatar","Purbasthali Dakshin",
    "Purbasthali Uttar","Katwa","Ketugram",
    "Mangalkot","Ausgram","Galsi","Pandabeswar",
    "Durgapur Purba","Durgapur Paschim",
    "Raniganj","Jamuria","Asansol Dakshin",
    "Asansol Uttar","Kulti","Barabani",
    "Dubrajpur","Suri","Bolpur","Nanoor",
    "Labhpur","Sainthia","Mayureswar",
    "Rampurhat","Hansan","Nalhati","Murarai"
]

TOTAL_SEATS = len(constituencies)

# ==========================================================
# RANDOM SEED
# ==========================================================

np.random.seed(42)

# ==========================================================
# CREATE DATA
# ==========================================================

data = []

for seat in constituencies:

    tmc_prev_vote = np.random.normal(48, 5)
    bjp_prev_vote = np.random.normal(38, 5)

    turnout_change = np.random.normal(8, 2)

    minority_population = np.random.uniform(5, 65)

    rural_population = np.random.uniform(20, 90)

    anti_incumbency = np.random.uniform(0, 10)

    tmc_candidate_popularity = np.random.uniform(40, 90)
    bjp_candidate_popularity = np.random.uniform(40, 90)

    previous_voters = np.random.randint(150000, 350000)

    deleted_voters = np.random.randint(1000, 12000)

    new_voters = np.random.randint(2000, 15000)

    voter_roll_impact = deleted_voters / previous_voters

    swing_factor = np.random.normal(0, 4)

    local_variation = np.random.uniform(-3, 3)

    # ======================================================
    # SCORE CALCULATION
    # ======================================================

    tmc_score = (
        tmc_prev_vote
        + (minority_population * 0.08)
        + (rural_population * 0.02)
        + (tmc_candidate_popularity * 0.05)
        - anti_incumbency
        - (turnout_change * 0.20)
        - (voter_roll_impact * 50)
        + swing_factor
        + local_variation
    )

    bjp_score = (
        bjp_prev_vote
        + ((100 - minority_population) * 0.04)
        + (turnout_change * 0.40)
        + (bjp_candidate_popularity * 0.05)
        + anti_incumbency
        + (voter_roll_impact * 40)
        - swing_factor
        - local_variation
    )

    winner = 1 if tmc_score > bjp_score else 0

    data.append([
        seat,
        tmc_prev_vote,
        bjp_prev_vote,
        turnout_change,
        minority_population,
        rural_population,
        anti_incumbency,
        tmc_candidate_popularity,
        bjp_candidate_popularity,
        deleted_voters,
        new_voters,
        voter_roll_impact,
        swing_factor,
        local_variation,
        winner
    ])

# ==========================================================
# DATAFRAME
# ==========================================================

columns = [

    "Constituency",
    "tmc_prev_vote",
    "bjp_prev_vote",
    "turnout_change",
    "minority_population",
    "rural_population",
    "anti_incumbency",
    "tmc_candidate_popularity",
    "bjp_candidate_popularity",
    "deleted_voters",
    "new_voters",
    "voter_roll_impact",
    "swing_factor",
    "local_variation",
    "winner"
]

df = pd.DataFrame(data, columns=columns)

# ==========================================================
# FEATURES & TARGET
# ==========================================================

X = df.drop(["Constituency", "winner"], axis=1)

y = df["winner"]

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================================
# RANDOM FOREST MODEL
# ==========================================================

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=12,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================================================
# ACCURACY
# ==========================================================

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n===================================")
print("MODEL ACCURACY")
print("===================================")

print(f"Accuracy : {accuracy*100:.2f}%")

# ==========================================================
# FULL PREDICTION
# ==========================================================

full_predictions = model.predict(X)

df["Prediction"] = full_predictions

df["Predicted_Winner"] = df["Prediction"].map({
    1: "TMC",
    0: "BJP"
})

# ==========================================================
# WIN PROBABILITY
# ==========================================================

probabilities = model.predict_proba(X)

# ==========================================================
# ALL 294 SEAT OUTPUT
# ==========================================================

print("\n===================================")
print("ALL 294 SEAT PREDICTIONS")
print("===================================")

for index, row in df.iterrows():

    constituency = row["Constituency"]

    winner = row["Predicted_Winner"]

    tmc_probability = probabilities[index][1] * 100

    bjp_probability = probabilities[index][0] * 100

    print(
        f"{constituency:30} | "
        f"Winner: {winner:3} | "
        f"TMC: {tmc_probability:.2f}% | "
        f"BJP: {bjp_probability:.2f}%"
    )

# ==========================================================
# FINAL SEAT COUNT
# ==========================================================

tmc_seats = np.sum(full_predictions == 1)

bjp_seats = np.sum(full_predictions == 0)

print("\n===================================")
print("FINAL SEAT COUNT")
print("===================================")

print(f"TMC Seats : {tmc_seats}")

print(f"BJP Seats : {bjp_seats}")

# ==========================================================
# FINAL RESULT
# ==========================================================

majority_mark = 148

print("\n===================================")
print("FINAL RESULT")
print("===================================")

if tmc_seats >= majority_mark:

    print("Prediction : TMC likely to form government")

elif bjp_seats >= majority_mark:

    print("Prediction : BJP likely to form government")

else:

    print("Prediction : Hung Assembly possible")

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

importance_df = pd.DataFrame({

    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n===================================")
print("FEATURE IMPORTANCE")
print("===================================")

print(importance_df)

# ==========================================================
# MONTE CARLO SIMULATION
# ==========================================================

SIMULATIONS = 5000

tmc_majority_count = 0
bjp_majority_count = 0
hung_count = 0

for sim in range(SIMULATIONS):

    sim_tmc = 0
    sim_bjp = 0

    for i in range(TOTAL_SEATS):

        tmc_random = random.uniform(-3, 3)

        bjp_random = random.uniform(-3, 3)

        tmc_final = probabilities[i][1] + tmc_random

        bjp_final = probabilities[i][0] + bjp_random

        if tmc_final > bjp_final:

            sim_tmc += 1

        else:

            sim_bjp += 1

    if sim_tmc >= majority_mark:

        tmc_majority_count += 1

    elif sim_bjp >= majority_mark:

        bjp_majority_count += 1

    else:

        hung_count += 1

# ==========================================================
# FINAL PROBABILITY
# ==========================================================

tmc_majority_probability = (
    tmc_majority_count / SIMULATIONS
) * 100

bjp_majority_probability = (
    bjp_majority_count / SIMULATIONS
) * 100

hung_probability = (
    hung_count / SIMULATIONS
) * 100

print("\n===================================")
print("MONTE CARLO SIMULATION")
print("===================================")

print(
    f"TMC Majority Probability : "
    f"{tmc_majority_probability:.2f}%"
)

print(
    f"BJP Majority Probability : "
    f"{bjp_majority_probability:.2f}%"
)

print(
    f"Hung Assembly Probability : "
    f"{hung_probability:.2f}%"
)

# ==========================================================
# SAVE CSV
# ==========================================================

df.to_csv(
    "west_bengal_294_seat_prediction.csv",
    index=False
)

print("\n===================================")
print("CSV SAVED")
print("===================================")

print("File : west_bengal_294_seat_prediction.csv")

print("\n===================================")
print("MODEL COMPLETED SUCCESSFULLY")
# Save trained model
# ==========================================================
# MODEL INFORMATION
# ==========================================================

model_data = {

    "developer":
    "Mohammed Juyel Haque",

    "model_name":
    "West Bengal 294 Seat Advanced Election ML Model",

    "algorithm":
    "RandomForestClassifier",

    "total_constituencies":
    TOTAL_SEATS,

    "features":
    list(X.columns),

    "trained_model":
    model
}

# ==========================================================
# SAVE MODEL
# ==========================================================

joblib.dump(
    model_data,
    "west_bengal_election_model.pkl"
)

print("Model saved successfully")

print(
    "Developer : Mohammed Juyel Haque"
)

print("===================================")
