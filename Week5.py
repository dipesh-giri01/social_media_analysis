import pandas as pd

 

# ─────────────────────────────────────────────

# TRAINING SET — social_media_data.csv

# ─────────────────────────────────────────────

train = pd.read_csv("social_media_data.csv")

print(f"Training raw: {train.shape}")

 

# Step 1 - Parse dates

train["date"] = pd.to_datetime(train["date"], errors="coerce")

 

# Step 2 - Remove invalid platforms

train = train[train["platform"].isin(["Twitter", "Instagram", "Facebook"])]

 

# Step 3 - Fill missing values with platform median

for col in ["likes", "shares", "comments", "views"]:

train[col] = train[col].fillna(train.groupby("platform")[col].transform("median"))

 

# Step 4 - Standardize text

train["platform"] = train["platform"].str.strip().str.title()

train["post_type"] = train["post_type"].str.strip().str.lower()

 

# Step 5 - Remove duplicates

train = train.drop_duplicates()

 

# Step 6 - Cap outliers

for col in ["likes", "shares", "comments", "views"]:

Q1, Q3 = train[col].quantile(0.25), train[col].quantile(0.75)

IQR = Q3 - Q1

train[col] = train[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)

 

# Step 7 - Encode categories

train["platform_encoded"] = train["platform"].map({"Twitter": 0, "Instagram": 1, "Facebook": 2})

train["post_type_encoded"] = train["post_type"].map({"image": 0, "video": 1, "text": 2})

 

# Step 8 - Extract time features

train["year"] = train["date"].dt.year

train["month"] = train["date"].dt.month

train["day_of_week"] = train["date"].dt.dayofweek

train["week_of_year"] = train["date"].dt.isocalendar().week.astype(int)

 

# Step 9 - Sort by date

train = train.sort_values("date").reset_index(drop=True)

 

train.to_csv("train_cleaned.csv", index=False)

print(f"Training clean: {train.shape} → saved as train_cleaned.csv")

 

 

# ─────────────────────────────────────────────

# TEST SET — cleaned_social_media_data.xlsx

# ─────────────────────────────────────────────

test = pd.read_excel("cleaned_social_media_data.xlsx")

print(f"\nTest raw: {test.shape}")

 

# Step 1 - Parse Publish_time

test["Publish_time"] = pd.to_datetime(test["Publish_time"], format="%m/%d/%Y %H:%M", errors="coerce")

test["date"] = pd.to_datetime(test["Publish_time"].dt.date)

 

# Step 2 - Drop fully null columns

test = test.drop(columns=test.columns[test.isnull().mean() == 1.0].tolist())

 

# Step 3 - Fill missing Follows with median

test["Follows"] = test["Follows"].fillna(test["Follows"].median())

 

# Step 4 - Fill missing Account_name

test["Account_name"] = test["Account_name"].fillna(test["Account_name"].dropna().iloc[0])

 

# Step 5 - Normalize Post_type labels

test["Post_type"] = test["Post_type"].str.strip().str.lower()

def normalize_post_type(pt):

if "reel" in pt or "video" in pt: return "video"

if "image" in pt or "carousel" in pt: return "image"

return "text"

test["post_type_normalized"] = test["Post_type"].apply(normalize_post_type)

 

# Step 6 - Encode categories

test["post_type_encoded"] = test["post_type_normalized"].map({"image": 0, "video": 1, "text": 2})

test["platform_encoded"] = 1 # all Instagram

 

# Step 7 - Extract time features

test["year"] = test["Publish_time"].dt.year

test["month"] = test["Publish_time"].dt.month

test["day_of_week"] = test["Publish_time"].dt.dayofweek

test["week_of_year"] = test["Publish_time"].dt.isocalendar().week.astype(int)

test["hour"] = test["Publish_time"].dt.hour

 

# Step 8 - Drop identifier columns

test = test.drop(columns=["Post_ID", "Account_ID", "Account_username", "Account_name", "Permalink", "Description"], errors="ignore")

 

# Step 9 - Remove duplicates

test = test.drop_duplicates()

 

# Step 10 - Sort by date

test = test.sort_values("Publish_time").reset_index(drop=True)

 

test.to_csv("test_cleaned.csv", index=False)

print(f"Test clean: {test.shape} → saved as test_cleaned.csv")

 

print("\nDone. Both files are ready.")
