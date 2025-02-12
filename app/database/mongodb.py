from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import os

load_dotenv()

class MongoDB:
    def __init__(self):
        self.client = None
        self.db = None

    async def connect_to_mongodb(self):
        self.client = AsyncIOMotorClient(os.getenv("MONGODB_URL"))
        self.db = self.client[os.getenv("MONGODB_DB_NAME", "rinova")]
        # Create indexes here later
        print("Connected to MongoDB!")

    async def close_mongodb_connection(self):
        if self.client:
            self.client.close()
            print("MongoDB connection closed.")

    @property
    def medical_notes(self):
        return self.db.medical_notes

    @property
    def extractions(self):
        return self.db.extractions

# Create a global instance
db = MongoDB()