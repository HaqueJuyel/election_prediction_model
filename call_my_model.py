# ==========================================================
# LOAD SAVED WEST BENGAL ELECTION MODEL
# AND USE IT FOR PREDICTION
# ==========================================================

import joblib
import pandas as pd

# ==========================================================
# LOAD MODEL
# ==========================================================

model = joblib.load(
    "west_bengal_election_model.pkl"
)

print("===================================")
print("MODEL LOADED SUCCESSFULLY")
print("===================================")

# ==========================================================
# CREATE INPUT DATA
# ==========================================================
#
# You can change these values
# according to any constituency
#
# ==========================================================

input_data = pd.DataFrame([{

    "tmc_prev_vote": 48,
    "bjp_prev_vote": 39,

    "turnout_change": 8,

    "minority_population": 32,

    "rural_population": 61,

    "anti_incumbency": 4,

    "tmc_candidate_popularity": 72,

    "bjp_candidate_popularity": 66,

    "deleted_voters": 4500,

    "new_voters": 9000,

    "voter_roll_impact": 0.02,

    "swing_factor": 1.5,

    "local_variation": 0.7

}])

# ==========================================================
# MAKE PREDICTION
# ==========================================================

prediction = model.predict(input_data)

# ==========================================================
# GET WIN PROBABILITY
# ==========================================================

probability = model.predict_proba(input_data)

tmc_probability = probability[0][1] * 100

bjp_probability = probability[0][0] * 100

# ==========================================================
# PRINT RESULT
# ==========================================================

print("\n===================================")
print("PREDICTION RESULT")
print("===================================")

if prediction[0] == 1:

    print("Predicted Winner : TMC")

else:

    print("Predicted Winner : BJP")

print(
    f"\nTMC Winning Probability : "
    f"{tmc_probability:.2f}%"
)

print(
    f"BJP Winning Probability : "
    f"{bjp_probability:.2f}%"
)

# ==========================================================
# OPTIONAL:
# SAVE RESULT TO CSV
# ==========================================================

result_df = pd.DataFrame({

    "Predicted_Winner": [
        "TMC" if prediction[0] == 1 else "BJP"
    ],

    "TMC_Probability": [
        round(tmc_probability, 2)
    ],

    "BJP_Probability": [
        round(bjp_probability, 2)
    ]
})

result_df.to_csv(
    "prediction_result.csv",
    index=False
)

print("\n===================================")
print("RESULT SAVED")
print("===================================")

print("File : prediction_result.csv")
