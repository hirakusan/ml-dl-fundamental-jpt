"""
Fundamental ML/DL Project
Demonstrasi konsep dasar Machine Learning dan Deep Learning
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, load_iris, load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

class FundamentalMLDemo:
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_sample_data(self):
        """Load and prepare sample dataset"""
        print("=== Loading Sample Data ===")
        
        # Generate synthetic dataset
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=10,
            n_redundant=5,
            n_clusters_per_class=1,
            random_state=42
        )
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        print(f"Classes: {np.unique(y)}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def traditional_ml_demo(self):
        """Demonstrate traditional ML algorithms"""
        print("\n=== Traditional Machine Learning Demo ===")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        
        # Models to compare
        models = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train_scaled, self.y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(self.y_test, y_pred)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred
            }
            
            print(f"{name} Accuracy: {accuracy:.4f}")
        
        # Plot comparison
        self.plot_model_comparison(results)
        
        return results
    
    def deep_learning_demo(self):
        """Demonstrate Deep Learning with TensorFlow/Keras"""
        print("\n=== Deep Learning Demo ===")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        
        # Build neural network
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print("Neural Network Architecture:")
        model.summary()
        
        # Train model
        print("\nTraining Neural Network...")
        history = model.fit(
            X_train_scaled, self.y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        # Evaluate model
        test_loss, test_accuracy = model.evaluate(X_test_scaled, self.y_test, verbose=0)
        print(f"Neural Network Test Accuracy: {test_accuracy:.4f}")
        
        # Plot training history
        self.plot_training_history(history)
        
        return model, history
    
    def plot_model_comparison(self, results):
        """Plot comparison of different models"""
        plt.figure(figsize=(12, 5))
        
        # Accuracy comparison
        plt.subplot(1, 2, 1)
        names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in names]
        
        bars = plt.bar(names, accuracies, color=['skyblue', 'lightcoral'])
        plt.title('Model Accuracy Comparison')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # Confusion matrix for best model
        plt.subplot(1, 2, 2)
        best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
        y_pred = results[best_model]['predictions']
        cm = confusion_matrix(self.y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {best_model}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        plt.tight_layout()
        plt.show()
    
    def plot_training_history(self, history):
        """Plot training history for neural network"""
        plt.figure(figsize=(12, 4))
        
        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def feature_importance_analysis(self, rf_model):
        """Analyze feature importance from Random Forest"""
        print("\n=== Feature Importance Analysis ===")
        
        # Get feature importance
        importance = rf_model.feature_importances_
        feature_names = [f'Feature_{i}' for i in range(len(importance))]
        
        # Create DataFrame
        feature_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Plot top 10 features
        plt.figure(figsize=(10, 6))
        top_features = feature_df.head(10)
        
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 10 Most Important Features')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
        return feature_df

def main():
    """Main function to run the demonstration"""
    print("🚀 Fundamental ML/DL Demo Starting...")
    
    # Initialize demo
    demo = FundamentalMLDemo()
    
    # Load data
    X_train, X_test, y_train, y_test = demo.load_sample_data()
    
    # Traditional ML demo
    ml_results = demo.traditional_ml_demo()
    
    # Deep Learning demo
    dl_model, history = demo.deep_learning_demo()
    
    # Feature importance analysis
    rf_model = ml_results['Random Forest']['model']
    feature_df = demo.feature_importance_analysis(rf_model)
    
    print("\n✅ Demo completed successfully!")
    print("\nKey Takeaways:")
    print("1. Traditional ML algorithms are interpretable and fast")
    print("2. Deep Learning can capture complex patterns")
    print("3. Feature engineering and scaling are crucial")
    print("4. Model evaluation requires multiple metrics")
    
    return demo, ml_results, dl_model

if __name__ == "__main__":
    demo, ml_results, dl_model = main()