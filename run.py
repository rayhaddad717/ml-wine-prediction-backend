from app import app
from dotenv import load_dotenv
import os
if __name__ == '__main__':
    load_dotenv()  # Load environment variables from .env file
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
