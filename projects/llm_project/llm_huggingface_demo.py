"""
LLM Project dengan Hugging Face
Demonstrasi penggunaan Large Language Models menggunakan Hugging Face Transformers
"""

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification,
    pipeline, Trainer, TrainingArguments, DataCollatorWithPadding
)
from datasets import Dataset, load_dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class LLMDemo:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
    def text_generation_demo(self):
        """Demonstrate text generation with GPT-2"""
        print("=== Text Generation Demo ===")
        
        # Load pre-trained GPT-2 model
        model_name = "gpt2"
        print(f"Loading {model_name} model...")
        
        # Create text generation pipeline
        generator = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name,
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Example prompts
        prompts = [
            "Artificial intelligence is revolutionizing",
            "The future of machine learning will be",
            "Deep learning applications in healthcare include"
        ]
        
        print("\nGenerating text for different prompts:")
        results = []
        
        for prompt in prompts:
            print(f"\nPrompt: {prompt}")
            
            # Generate text
            outputs = generator(
                prompt,
                max_length=100,
                num_return_sequences=2,
                temperature=0.7,
                pad_token_id=generator.tokenizer.eos_token_id
            )
            
            for i, output in enumerate(outputs):
                generated_text = output['generated_text']
                print(f"Generation {i+1}: {generated_text}")
                results.append({
                    'prompt': prompt,
                    'generated_text': generated_text
                })
        
        return results
    
    def sentiment_analysis_demo(self):
        """Demonstrate sentiment analysis"""
        print("\n=== Sentiment Analysis Demo ===")
        
        # Create sentiment analysis pipeline
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Example texts
        texts = [
            "I love this new machine learning framework!",
            "This model is terrible and doesn't work at all.",
            "The performance is okay, nothing special.",
            "Amazing breakthrough in AI technology!",
            "I'm disappointed with the results."
        ]
        
        print("Analyzing sentiment for different texts:")
        results = []
        
        for text in texts:
            result = sentiment_pipeline(text)
            sentiment = result[0]['label']
            confidence = result[0]['score']
            
            print(f"Text: {text}")
            print(f"Sentiment: {sentiment} (Confidence: {confidence:.3f})")
            print("-" * 50)
            
            results.append({
                'text': text,
                'sentiment': sentiment,
                'confidence': confidence
            })
        
        # Visualize results
        self.plot_sentiment_analysis(results)
        
        return results
    
    def question_answering_demo(self):
        """Demonstrate question answering"""
        print("\n=== Question Answering Demo ===")
        
        # Create QA pipeline
        qa_pipeline = pipeline(
            "question-answering",
            model="distilbert-base-cased-distilled-squad",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Context and questions
        context = """
        Machine learning is a subset of artificial intelligence that focuses on algorithms 
        that can learn from data. Deep learning is a subset of machine learning that uses 
        neural networks with multiple layers. Popular deep learning frameworks include 
        TensorFlow, PyTorch, and Keras. These frameworks make it easier to build and 
        train complex neural networks for various tasks such as image recognition, 
        natural language processing, and speech recognition.
        """
        
        questions = [
            "What is machine learning?",
            "What are some popular deep learning frameworks?",
            "What tasks can neural networks perform?",
            "How is deep learning related to machine learning?"
        ]
        
        print("Answering questions based on context:")
        results = []
        
        for question in questions:
            result = qa_pipeline(question=question, context=context)
            answer = result['answer']
            confidence = result['score']
            
            print(f"Question: {question}")
            print(f"Answer: {answer}")
            print(f"Confidence: {confidence:.3f}")
            print("-" * 50)
            
            results.append({
                'question': question,
                'answer': answer,
                'confidence': confidence
            })
        
        return results
    
    def text_classification_training_demo(self):
        """Demonstrate fine-tuning for text classification"""
        print("\n=== Text Classification Training Demo ===")
        
        # Create sample dataset
        sample_data = {
            'text': [
                "This movie is amazing!", "I hate this film", "Great acting and plot",
                "Terrible storyline", "Best movie ever", "Waste of time",
                "Excellent cinematography", "Poor dialogue", "Love the characters",
                "Boring and predictable", "Fantastic performance", "Not recommended",
                "Beautiful scenes", "Disappointing ending", "Must watch film",
                "Overrated movie", "Brilliant direction", "Confusing plot",
                "Outstanding soundtrack", "Weak script"
            ],
            'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        }
        
        # Create dataset
        dataset = Dataset.from_dict(sample_data)
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        train_dataset = dataset.select(range(train_size))
        eval_dataset = dataset.select(range(train_size, len(dataset)))
        
        # Load model and tokenizer
        model_name = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
        
        # Tokenize datasets
        def tokenize_function(examples):
            return tokenizer(examples['text'], truncation=True, padding=True)
        
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        eval_dataset = eval_dataset.map(tokenize_function, batched=True)
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=10,
            weight_decay=0.01,
            logging_dir="./logs",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        
        # Define compute metrics function
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return {'accuracy': accuracy_score(labels, predictions)}
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        
        print("Training model...")
        # Train model (commented out for demo - would take time)
        # trainer.train()
        
        print("Training completed! (Demo mode - actual training skipped)")
        
        return trainer, model, tokenizer
    
    def plot_sentiment_analysis(self, results):
        """Plot sentiment analysis results"""
        df = pd.DataFrame(results)
        
        plt.figure(figsize=(12, 6))
        
        # Sentiment distribution
        plt.subplot(1, 2, 1)
        sentiment_counts = df['sentiment'].value_counts()
        plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
        plt.title('Sentiment Distribution')
        
        # Confidence scores
        plt.subplot(1, 2, 2)
        colors = ['red' if s == 'NEGATIVE' else 'blue' if s == 'NEUTRAL' else 'green' 
                 for s in df['sentiment']]
        plt.bar(range(len(df)), df['confidence'], color=colors, alpha=0.7)
        plt.xlabel('Text Index')
        plt.ylabel('Confidence Score')
        plt.title('Confidence Scores by Sentiment')
        plt.xticks(range(len(df)), [f'Text {i+1}' for i in range(len(df))], rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def model_comparison_demo(self):
        """Compare different model sizes and architectures"""
        print("\n=== Model Comparison Demo ===")
        
        models_to_compare = [
            ("distilbert-base-uncased", "DistilBERT"),
            ("bert-base-uncased", "BERT-Base"),
            ("roberta-base", "RoBERTa")
        ]
        
        sample_text = "This is a great example of natural language processing!"
        
        results = []
        
        for model_name, display_name in models_to_compare:
            print(f"\nTesting {display_name}...")
            
            try:
                # Create classifier pipeline
                classifier = pipeline(
                    "sentiment-analysis",
                    model=model_name,
                    device=0 if torch.cuda.is_available() else -1
                )
                
                # Get prediction
                result = classifier(sample_text)
                
                # Get model size (approximate)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                
                num_params = sum(p.numel() for p in model.parameters())
                
                results.append({
                    'model': display_name,
                    'sentiment': result[0]['label'],
                    'confidence': result[0]['score'],
                    'parameters': num_params
                })
                
                print(f"Sentiment: {result[0]['label']}, Confidence: {result[0]['score']:.3f}")
                print(f"Parameters: {num_params:,}")
                
            except Exception as e:
                print(f"Error with {display_name}: {str(e)}")
        
        # Plot comparison
        if results:
            self.plot_model_comparison(results)
        
        return results
    
    def plot_model_comparison(self, results):
        """Plot model comparison results"""
        df = pd.DataFrame(results)
        
        plt.figure(figsize=(15, 5))
        
        # Confidence comparison
        plt.subplot(1, 3, 1)
        plt.bar(df['model'], df['confidence'])
        plt.title('Model Confidence Comparison')
        plt.ylabel('Confidence Score')
        plt.xticks(rotation=45)
        
        # Parameter count
        plt.subplot(1, 3, 2)
        plt.bar(df['model'], df['parameters'])
        plt.title('Model Size Comparison')
        plt.ylabel('Number of Parameters')
        plt.xticks(rotation=45)
        
        # Efficiency (confidence/parameters ratio)
        plt.subplot(1, 3, 3)
        efficiency = df['confidence'] / (df['parameters'] / 1e6)  # per million parameters
        plt.bar(df['model'], efficiency)
        plt.title('Efficiency (Confidence/Million Parameters)')
        plt.ylabel('Efficiency Score')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()

def main():
    """Main function to run LLM demonstrations"""
    print("🤖 LLM Demo with Hugging Face Starting...")
    
    # Initialize demo
    demo = LLMDemo()
    
    # Text generation demo
    generation_results = demo.text_generation_demo()
    
    # Sentiment analysis demo
    sentiment_results = demo.sentiment_analysis_demo()
    
    # Question answering demo
    qa_results = demo.question_answering_demo()
    
    # Text classification training demo
    trainer, model, tokenizer = demo.text_classification_training_demo()
    
    # Model comparison demo
    comparison_results = demo.model_comparison_demo()
    
    print("\n✅ LLM Demo completed successfully!")
    print("\nKey Takeaways:")
    print("1. Hugging Face provides easy-to-use pre-trained models")
    print("2. Pipelines simplify common NLP tasks")
    print("3. Fine-tuning allows customization for specific tasks")
    print("4. Model size vs performance trade-offs are important")
    print("5. Different models excel at different tasks")
    
    return demo, generation_results, sentiment_results, qa_results

if __name__ == "__main__":
    demo, generation_results, sentiment_results, qa_results = main()