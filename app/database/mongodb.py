from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import os

load_dotenv()

class MongoDB:
    client: AsyncIOMotorClient = None
    db = None

    async def connect_to_mongodb(self):
        self.client = AsyncIOMotorClient(os.getenv("MONGODB_URL"))
        self.db = self.client[os.getenv("MONGODB_DB_NAME", "rinova")]
        
        # Create indexes for better query performance
        await self.db.medical_notes.create_index("date")
        await self.db.medical_notes.create_index([("doctor_name", 1)])
        await self.db.medical_notes.create_index([("patient_name", 1)])
        
        print("Connected to MongoDB!")

    async def close_mongodb_connection(self):
        if self.client:
            self.client.close()
            print("MongoDB connection closed.")

    def get_db(self):
        return self.db

# Create a global instance
db = MongoDB()

# Function to get database instance
def get_database():
    return db.get_db()
