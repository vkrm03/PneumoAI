from pymongo import MongoClient

try:
    client = MongoClient("mongodb://127.0.0.1:27017/", serverSelectionTimeoutMS=5000)

    # Force a check to see if connection works
    client.server_info()
    print("✅ Connected to MongoDB!")

    # Create or connect to a database
    db = client["pneumodetect_db"]

    # Create or connect to a collection
    users_col = db["users"]

    # Insert a test document
    test_user = {"email": "test@test.com", "password": "123"}
    result = users_col.insert_one(test_user)
    print("Inserted document ID:", result.inserted_id)

    # Fetch the inserted document
    user = users_col.find_one({"email": "test@test.com"})
    print("Found user in DB:", user)

except Exception as e:
    print("❌ MongoDB Connection Error:", e)
