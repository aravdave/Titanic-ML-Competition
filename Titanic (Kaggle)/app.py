import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

def change_sex_to_binary(table):
    for sex in table["Sex"]:
        if sex == "female":
            sex = 1
        else:
            sex = 0

change_sex_to_binary(train_data)
change_sex_to_binary(test_data)
my_imputer = SimpleImputer()

features = ["Pclass", "Sex", "Age", "Parch", "Fare"]
train_X = train_data[features]
test_X = test_data[features]

# train_X = pd.get_dummies(train_data[features])
# test_X = pd.get_dummies(test_data[features])


imputed_train_X = pd.DataFrame(my_imputer.fit_transform(train_X))
imputed_test_X = pd.DataFrame(my_imputer.transform(test_X))
imputed_test_X.columns = test_X.columns
imputed_train_X.columns = train_X.columns
train_y = train_data.Survived

model = RandomForestClassifier(n_estimators=100, random_state=1)
model.fit(imputed_train_X,train_y)
predictions = model.predict(test_X)

output = pd.DataFrame({"PassengerId":test_data.PassengerId, "Survived": predictions})
output.to_csv("submission.csv", index=False)
print("Executed w/o error")