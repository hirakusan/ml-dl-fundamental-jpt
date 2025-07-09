# ml-dl-fundamental-jpt

ML/DL Fundamental Workshop - Docker Edition
Proyek ini dibuat untuk seminar dan workshop fundamental Machine Learning & Deep Learning menggunakan Docker, ditujukan untuk mahasiswa, dosen, dan peneliti yang ingin memahami konsep dasar ML/DL dengan implementasi praktis.
📋 Deskripsi Project
Repository ini berisi materi lengkap workshop ML/DL fundamental yang mencakup:

Setup environment ML/DL menggunakan Docker
Implementasi algoritma ML/DL dasar
Pengembangan model LLM sederhana dengan HuggingFace
Contoh project praktis dan hands-on

🚀 Fitur Utama
1. Docker Environment Setup

Dockerfile yang sudah dikonfigurasi dengan library ML/DL terlengkap
Versi library yang stabil dan direkomendasikan
Environment yang reproducible untuk semua peserta

2. Fundamental ML/DL Projects

Classification: Implementasi algoritma klasifikasi (SVM, Random Forest, Neural Networks)
Regression: Model prediksi dengan Linear/Polynomial Regression
Clustering: K-Means dan Hierarchical Clustering
Deep Learning: CNN untuk image classification, RNN untuk sequence data
Data Preprocessing: Feature engineering dan data cleaning

3. LLM Development dengan HuggingFace

Fine-tuning pre-trained models
Text generation dan sentiment analysis
Question-answering system
Custom tokenizer dan model training

🛠️ Tech Stack

Python: 3.9+
ML Libraries: scikit-learn, pandas, numpy, matplotlib, seaborn
DL Frameworks: PyTorch, TensorFlow/Keras
LLM: HuggingFace Transformers, Datasets, Tokenizers
Visualization: Plotly, Jupyter Notebook
Container: Docker, Docker Compose

📦 Instalasi
Prerequisites

Docker Desktop
Git

Quick Start
bash# Clone repository
git clone https://github.com/yourusername/ml-dl-fundamental-workshop.git
cd ml-dl-fundamental-workshop

# Build Docker image
docker-compose build

# Run container
docker-compose up
Akses Jupyter Notebook di: http://localhost:8888
📚 Struktur Project
ml-dl-fundamental-workshop/
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements.txt
├── notebooks/
│   ├── 01_ml_basics/
│   │   ├── classification.ipynb
│   │   ├── regression.ipynb
│   │   └── clustering.ipynb
│   ├── 02_deep_learning/
│   │   ├── cnn_image_classification.ipynb
│   │   ├── rnn_sequence_modeling.ipynb
│   │   └── autoencoder.ipynb
│   └── 03_llm_huggingface/
│       ├── text_generation.ipynb
│       ├── sentiment_analysis.ipynb
│       └── question_answering.ipynb
├── src/
│   ├── data_preprocessing/
│   ├── models/
│   └── utils/
├── data/
│   ├── raw/
│   └── processed/
└── README.md
🎯 Learning Objectives
Setelah mengikuti workshop ini, peserta akan mampu:

Memahami konsep fundamental ML/DL
Mengimplementasikan algoritma ML/DL dari scratch
Menggunakan Docker untuk reproducible ML environment
Mengembangkan model LLM dengan HuggingFace
Menerapkan best practices dalam ML/DL development

📖 Materi Workshop
Session 1: ML Fundamentals

Introduction to Machine Learning
Data Preprocessing & Feature Engineering
Supervised Learning (Classification & Regression)
Unsupervised Learning (Clustering)
Model Evaluation & Validation

Session 2: Deep Learning Basics

Neural Networks from Scratch
Convolutional Neural Networks (CNN)
Recurrent Neural Networks (RNN)
Transfer Learning
Hyperparameter Tuning

Session 3: LLM dengan HuggingFace

Introduction to Transformers
Pre-trained Models & Fine-tuning
Text Generation & GPT Models
BERT untuk NLP Tasks
Custom Model Development

🎓 Target Audience

Mahasiswa: S1/S2 Informatika, Data Science, AI
Dosen: Yang ingin mengajar ML/DL dengan pendekatan praktis
Peneliti: Yang membutuhkan foundation ML/DL untuk research
Praktisi: Yang ingin memahami fundamental sebelum advanced topics

📝 Prerequisites

Pemahaman dasar Python
Konsep matematika: Linear Algebra, Statistics
Familiar dengan command line
Basic understanding of Docker (optional, akan dijelaskan)

🤝 Contributing
Kontribusi sangat diterima! Silakan:

Fork repository
Buat feature branch (git checkout -b feature/amazing-feature)
Commit changes (git commit -m 'Add amazing feature')
Push ke branch (git push origin feature/amazing-feature)
Buat Pull Request

📧 Contact

Instructor: HIBA AMBARA
LinkedIn: https://www.linkedin.com/in/hiba-ambara-5b7213204/
telegram : https://t.me/hirakusan

📄 License
Project ini dilisensikan under MIT License - lihat file LICENSE untuk detail.
🙏 Acknowledgments

HuggingFace untuk amazing transformers library 🚀
Kaggle untuk datasets
Docker community untuk containerization best practices
Open source ML/DL community


Happy Learning! 🚀
"The best way to learn ML/DL is by doing it hands-on"