## 🎶SISE'zam - Tailored tracks for your data-enhanced listening
**Music recommendation app**
SISE'zam is a music recognition application that goes beyond simple track identification. Leveraging machine learning technologies, it analyzes the genre and lyrics of nearly 29000 songs to generate personalized recommendations. Our algorithm is using cutting-edge techniques such as vector embeddings, clustering, APIs and Natural Language Processing to provide you with the best recommendations.

## Table of Contents
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [Technical Details](#technical-details)
- [Contributing](#contributing)
- [Authors](#authors)  

## 🔍 Key Features
- **💿 Track upload**: drag and drop various music file formats for automated recognition (mp3, flac, ogg, wav and m4a)
- **🎤 On-the-fly recognition**: Using your microphone, sample a 5-second clip and get your song recognized in a matter of seconds!
- **📊 Advanced Data Visualization**: Interactive charts and graphs to understand track classification

## 🚀 Installation
### Prerequisites
- Docker installed on your system
- adding a secrets.toml file under the repository .streamlit with your GENIUS_TOKEN
- deactivate Windows microphone enhancement if you're planning to use on-the-fly recognition.

### Quick Start with Docker Image
If you have a prebuilt "Challenge_Web_Mining.tar" Docker image, you can load and run it as follows:
1. Load the image:
   ```bash
   docker load -i Challenge_Web_Mining.tar
   ```
2. Verify the image name (usually "Challenge_Web_Mining") by running:
   ```bash
   docker images
   ```
3. Run the container:
   ```bash
   docker run -p 8501:8501 Challenge_Web_Mining
   ```
4. Navigate to:
   ```
   http://localhost:8501
   ```

### Quick Start with Docker
1. Clone the repository:
   ```bash
   git clone https://github.com/karamoko17/Challenge_Web_Mining.git
   cd Challenge_Web_Mining
   ```
2. Build the Docker image:
   ```bash
   docker build -t Challenge_Web_Mining .
   ```
3. Run the container:
   ```bash
   docker run -p 8501:8501 Challenge_Web_Mining
   ```
4. Open your browser and navigate to:
   ```
   http://localhost:8501
   ```
### Manual Installation
If you prefer not to use Docker:
1. Clone the repository:
   ```bash
   git clone https://github.com/karamoko17/Challenge_Web_Mining.git
   cd Challenge_Web_Mining
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run Streamlit/app/app.py
   ```
   
## 📝 Usage Guide
1. **Playlist generation**: Upload your music file, or sample a clip with your computer's microphone
2. **Embedding analysis**: Explore our database
3. **Enjoy the music!**

## 🛠️ Technical Details
SISE'zam is built with:
- **Python 3.11+**
- **Streamlit** for the web interface
- **ChromaDB** for embedding database
- **Plotly** for interactive visualizations
- **scikit-learn** for machine learning capabilities
  
## 🤝 Contributing
Contributions are welcome! To contribute:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

## 👥 Authors

- [Awa KARAMOKO](https://github.com/karamoko17)
- [Bertrand KLEIN](https://github.com/bertrandklein)
- [Antoine ORUEZABALA](https://github.com/AntoineORUEZABALA)
- [Béranger THOMAS](https://github.com/berangerthomas)
