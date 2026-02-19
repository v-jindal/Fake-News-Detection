import pandas as pd
import numpy as np
import re
import warnings
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pygame
from gtts import gTTS
import tempfile
import os
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

warnings.filterwarnings('ignore')

class ProfessionalNewsDetector:
    def __init__(self):
        self.models = {}
        self.vectorizer = None
        self.ensemble_model = None
        self.root = None
        self.data = None
        self.X_test = None
        self.y_test = None
        self.setup_gui()
        
    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Fake News Detection System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f8f9fa')
        self.center_window(1200, 800)
        self.setup_styles()
        self.create_main_interface()
        
    def center_window(self, width, height):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('Title.TLabel', 
                       background='#2c3e50', 
                       foreground='white', 
                       font=('Arial', 18, 'bold'),
                       padding=10)
        
        style.configure('Subtitle.TLabel',
                       background='#2c3e50',
                       foreground='#bdc3c7',
                       font=('Arial', 11),
                       padding=5)
        
        style.configure('Card.TFrame',
                       background='white',
                       relief='raised',
                       borderwidth=1)
        
        style.configure('Primary.TButton',
                       background='#3498db',
                       foreground='white',
                       font=('Arial', 10, 'bold'),
                       padding=(15, 8))
        
        style.configure('Success.TButton',
                       background='#27ae60',
                       foreground='white',
                       font=('Arial', 10, 'bold'),
                       padding=(15, 8))
        
        style.configure('Warning.TButton',
                       background='#e67e22',
                       foreground='white',
                       font=('Arial', 10, 'bold'),
                       padding=(15, 8))
        
        style.configure('Danger.TButton',
                       background='#e74c3c',
                       foreground='white',
                       font=('Arial', 10, 'bold'),
                       padding=(15, 8))
        
        style.configure('Accent.TButton',
                       background='#9b59b6',
                       foreground='white',
                       font=('Arial', 10, 'bold'),
                       padding=(15, 8))
        
    def create_main_interface(self):
        # Header
        header_frame = ttk.Frame(self.root, style='Title.TFrame')
        header_frame.pack(fill='x', padx=0, pady=0)
        
        title_label = ttk.Label(header_frame, 
                               text="FAKE NEWS DETECTION SYSTEM", 
                               style='Title.TLabel')
        title_label.pack(fill='x')
        
        subtitle_label = ttk.Label(header_frame, 
                                  text="AI-Powered Content Verification Platform with Complete EDA & Model Evaluation", 
                                  style='Subtitle.TLabel')
        subtitle_label.pack(fill='x')
        
        # Main content area
        main_container = ttk.Frame(self.root, padding=20)
        main_container.pack(fill='both', expand=True)
        
        # Create notebook for tabs
        notebook = ttk.Notebook(main_container)
        notebook.pack(fill='both', expand=True)
        
        # Analysis Tab
        analysis_frame = ttk.Frame(notebook)
        notebook.add(analysis_frame, text="📊 News Analysis")
        
        # EDA Tab
        eda_frame = ttk.Frame(notebook)
        notebook.add(eda_frame, text="🔍 Data Analysis")
        
        # Models Tab
        models_frame = ttk.Frame(notebook)
        notebook.add(models_frame, text="🤖 Model Evaluation")
        
        self.create_analysis_tab(analysis_frame)
        self.create_eda_tab(eda_frame)
        self.create_models_tab(models_frame)
        self.create_status_bar()
        
    def create_analysis_tab(self, parent):
        # Create two-column layout
        left_frame = ttk.Frame(parent)
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        right_frame = ttk.Frame(parent)
        right_frame.pack(side='right', fill='both', expand=True, padx=(10, 0))
        
        self.create_input_section(left_frame)
        self.create_analysis_section(right_frame)
        self.create_control_panel(left_frame)
        
    def create_eda_tab(self, parent):
        eda_container = ttk.Frame(parent, padding=10)
        eda_container.pack(fill='both', expand=True)
        
        # EDA Controls
        eda_controls = ttk.Frame(eda_container, style='Card.TFrame', padding=10)
        eda_controls.pack(fill='x', pady=(0, 10))
        
        ttk.Label(eda_controls, 
                 text="📈 EXPLORATORY DATA ANALYSIS",
                 font=('Arial', 14, 'bold'),
                 foreground='#2c3e50').pack(anchor='w', pady=(0, 10))
        
        controls_frame = ttk.Frame(eda_controls)
        controls_frame.pack(fill='x')
        
        ttk.Button(controls_frame, 
                  text="📋 Dataset Summary", 
                  command=self.show_dataset_summary,
                  style='Primary.TButton').pack(side='left', padx=(0, 5))
        
        ttk.Button(controls_frame, 
                  text="📊 Descriptive Statistics", 
                  command=self.show_descriptive_stats,
                  style='Success.TButton').pack(side='left', padx=5)
        
        ttk.Button(controls_frame, 
                  text="📈 Generate Visualizations", 
                  command=self.generate_visualizations,
                  style='Warning.TButton').pack(side='left', padx=5)
        
        # EDA Results
        self.eda_text = scrolledtext.ScrolledText(eda_container, 
                                                 height=25, 
                                                 font=('Consolas', 9),
                                                 wrap='word',
                                                 padx=10,
                                                 pady=10,
                                                 bg='#f8f9fa',
                                                 relief='solid',
                                                 borderwidth=1)
        self.eda_text.pack(fill='both', expand=True)
        
    def create_models_tab(self, parent):
        models_container = ttk.Frame(parent, padding=10)
        models_container.pack(fill='both', expand=True)
        
        # Model Evaluation Section
        model_eval_frame = ttk.Frame(models_container, style='Card.TFrame', padding=15)
        model_eval_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(model_eval_frame, 
                 text="🤖 MODEL EVALUATION RESULTS",
                 font=('Arial', 14, 'bold'),
                 foreground='#2c3e50').pack(anchor='w', pady=(0, 10))
        
        eval_controls = ttk.Frame(model_eval_frame)
        eval_controls.pack(fill='x')
        
        ttk.Button(eval_controls, 
                  text="📊 Show All Model Evaluations", 
                  command=self.show_all_model_evaluations,
                  style='Success.TButton').pack(side='left', padx=(0, 5))
        
        ttk.Button(eval_controls, 
                  text="📈 Compare Models", 
                  command=self.compare_models,
                  style='Primary.TButton').pack(side='left', padx=5)
        
        # Model Results
        self.models_text = scrolledtext.ScrolledText(models_container, 
                                                   height=20, 
                                                   font=('Consolas', 9),
                                                   wrap='word',
                                                   padx=10,
                                                   pady=10,
                                                   bg='#f8f9fa',
                                                   relief='solid',
                                                   borderwidth=1)
        self.models_text.pack(fill='both', expand=True)
        
    def create_input_section(self, parent):
        input_card = ttk.Frame(parent, style='Card.TFrame', padding=15)
        input_card.pack(fill='both', expand=True, pady=(0, 10))
        
        section_header = ttk.Frame(input_card)
        section_header.pack(fill='x', pady=(0, 10))
        
        ttk.Label(section_header, 
                 text="📝 CONTENT INPUT",
                 font=('Arial', 14, 'bold'),
                 foreground='#2c3e50').pack(anchor='w')
        
        ttk.Label(input_card, 
                 text="Enter News Content:",
                 font=('Arial', 11, 'bold'),
                 foreground='#34495e').pack(anchor='w', pady=(0, 5))
        
        self.text_input = scrolledtext.ScrolledText(input_card, 
                                                   height=8, 
                                                   font=('Arial', 11),
                                                   wrap='word',
                                                   padx=10,
                                                   pady=10,
                                                   bg='#f8f9fa',
                                                   relief='solid',
                                                   borderwidth=1)
        self.text_input.pack(fill='both', expand=True)
        
        voice_frame = ttk.Frame(input_card)
        voice_frame.pack(fill='x', pady=(10, 0))
        
        ttk.Button(voice_frame, 
                  text="🎤 Voice Input", 
                  command=self.voice_input,
                  style='Accent.TButton').pack(side='left', padx=(0, 5))
        
        ttk.Button(voice_frame, 
                  text="📢 Speak Results", 
                  command=self.speak_result,
                  style='Warning.TButton').pack(side='left', padx=5)
        
    def create_analysis_section(self, parent):
        analysis_card = ttk.Frame(parent, style='Card.TFrame', padding=15)
        analysis_card.pack(fill='both', expand=True, pady=(0, 10))
        
        section_header = ttk.Frame(analysis_card)
        section_header.pack(fill='x', pady=(0, 10))
        
        ttk.Label(section_header, 
                 text="📊 ANALYSIS RESULTS",
                 font=('Arial', 12, 'bold'),
                 foreground='#2c3e50').pack(anchor='w')
        
        self.results_text = scrolledtext.ScrolledText(analysis_card, 
                                                     height=12, 
                                                     font=('Consolas', 10),
                                                     wrap='word',
                                                     padx=10,
                                                     pady=10,
                                                     bg='#f8f9fa',
                                                     relief='solid',
                                                     borderwidth=1)
        self.results_text.pack(fill='both', expand=True)
        
        self.confidence_frame = ttk.Frame(analysis_card)
        self.confidence_frame.pack(fill='x', pady=(10, 0))
        
    def create_control_panel(self, parent):
        control_card = ttk.Frame(parent, style='Card.TFrame', padding=15)
        control_card.pack(fill='x', pady=(10, 0))
        
        ttk.Label(control_card, 
                 text="⚙️ SYSTEM CONTROLS",
                 font=('Arial', 12, 'bold'),
                 foreground='#2c3e50').pack(anchor='w', pady=(0, 10))
        
        control_frame = ttk.Frame(control_card)
        control_frame.pack(fill='x')
        
        ttk.Button(control_frame, 
                  text="🚀 Initialize AI Models", 
                  command=self.train_models,
                  style='Success.TButton').pack(side='left', padx=(0, 5), fill='x', expand=True)
        
        ttk.Button(control_frame, 
                  text="🔍 Analyze Content", 
                  command=self.analyze_text,
                  style='Primary.TButton').pack(side='left', padx=5, fill='x', expand=True)
        
        ttk.Button(control_frame, 
                  text="🧹 Clear All", 
                  command=self.clear_all,
                  style='Danger.TButton').pack(side='left', padx=(5, 0), fill='x', expand=True)
        
    def create_status_bar(self):
        status_frame = ttk.Frame(self.root, relief='sunken', borderwidth=1)
        status_frame.pack(side='bottom', fill='x')
        
        self.status_var = tk.StringVar()
        self.status_var.set("System Ready - Initialize AI models to begin analysis")
        status_bar = ttk.Label(status_frame, 
                              textvariable=self.status_var, 
                              relief='sunken',
                              background='#ecf0f1',
                              foreground='#2c3e50',
                              font=('Arial', 9),
                              padding=5)
        status_bar.pack(fill='x')
        
    def print_to_console(self, message, target="eda"):
        """Print message to specified console"""
        print(message)
        if target == "eda":
            current_text = self.eda_text.get('1.0', tk.END)
            self.eda_text.delete('1.0', tk.END)
            self.eda_text.insert('1.0', current_text + message + '\n')
            self.eda_text.see(tk.END)
        elif target == "models":
            current_text = self.models_text.get('1.0', tk.END)
            self.models_text.delete('1.0', tk.END)
            self.models_text.insert('1.0', current_text + message + '\n')
            self.models_text.see(tk.END)
        
    def update_confidence_display(self, confidence):
        for widget in self.confidence_frame.winfo_children():
            widget.destroy()
            
        ttk.Label(self.confidence_frame, 
                 text="Confidence Level:",
                 font=('Arial', 10, 'bold'),
                 foreground='#2c3e50').pack(anchor='w')
        
        confidence_bar = ttk.Progressbar(self.confidence_frame, 
                                        orient='horizontal', 
                                        length=200, 
                                        mode='determinate',
                                        maximum=100)
        confidence_bar['value'] = confidence * 100
        confidence_bar.pack(fill='x', pady=(5, 0))
        
        confidence_color = '#27ae60' if confidence > 0.7 else '#e67e22' if confidence > 0.5 else '#e74c3c'
        confidence_label = ttk.Label(self.confidence_frame,
                                   text=f"{confidence:.1%}",
                                   font=('Arial', 11, 'bold'),
                                   foreground=confidence_color)
        confidence_label.pack(pady=(5, 0))
        
    def voice_input(self):
        self.status_var.set("Listening... Please speak the news content now")
        
        def listen():
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                try:
                    recognizer.adjust_for_ambient_noise(source, duration=1)
                    self.root.after(0, lambda: self.status_var.set("Listening... Speak now"))
                    audio = recognizer.listen(source, timeout=15, phrase_time_limit=30)
                    text = recognizer.recognize_google(audio)
                    self.root.after(0, self.update_voice_text, text)
                    self.root.after(0, lambda: self.status_var.set("Voice input captured successfully"))
                except sr.WaitTimeoutError:
                    self.root.after(0, lambda: self.status_var.set("No speech detected - timeout"))
                except sr.UnknownValueError:
                    self.root.after(0, lambda: self.status_var.set("Could not understand audio input"))
                except Exception as e:
                    error_msg = str(e)
                    self.root.after(0, lambda: self.status_var.set(f"Voice input error: {error_msg}"))
        
        threading.Thread(target=listen, daemon=True).start()
        
    def update_voice_text(self, text):
        self.text_input.delete('1.0', tk.END)
        self.text_input.insert('1.0', text)
        
    def speak_result(self):
        results_text = self.results_text.get('1.0', tk.END).strip()
        if not results_text:
            messagebox.showwarning("Warning", "No analysis results available. Please analyze content first.")
            return
            
        self.status_var.set("Generating audio summary...")
        
        def generate_speech():
            try:
                lines = results_text.split('\n')
                verdict = "Unknown"
                confidence = "Unknown"
                
                for line in lines:
                    if "FINAL VERDICT:" in line:
                        verdict_part = line.split("FINAL VERDICT:")[-1].strip()
                        verdict = re.sub(r'[^\w\s]', '', verdict_part).strip()
                    elif "CONFIDENCE LEVEL:" in line:
                        confidence_part = line.split("CONFIDENCE LEVEL:")[-1].strip()
                        confidence_match = re.search(r'(\d+\.?\d*)%', confidence_part)
                        if confidence_match:
                            confidence = confidence_match.group(1) + "%"
                
                if verdict == "Unknown":
                    for line in lines:
                        if "🔴 FINAL VERDICT:" in line or "🟢 FINAL VERDICT:" in line:
                            verdict_part = line.split("FINAL VERDICT:")[-1].strip()
                            verdict = re.sub(r'[^\w\s]', '', verdict_part).strip()
                        elif "🟢 CONFIDENCE LEVEL:" in line or "🟡 CONFIDENCE LEVEL:" in line or "🔴 CONFIDENCE LEVEL:" in line:
                            confidence_part = line.split("CONFIDENCE LEVEL:")[-1].strip()
                            confidence_match = re.search(r'(\d+\.?\d*)%', confidence_part)
                            if confidence_match:
                                confidence = confidence_match.group(1) + "%"
                
                if verdict == "Unknown":
                    if "FAKE NEWS" in results_text:
                        verdict = "Fake News"
                    elif "REAL NEWS" in results_text:
                        verdict = "Real News"
                
                if confidence == "Unknown":
                    confidence_match = re.search(r'(\d+\.?\d*)%', results_text)
                    if confidence_match:
                        confidence = confidence_match.group(1) + "%"
                    else:
                        confidence = "Unknown"
                
                speech_text = f"News analysis complete. Verdict: {verdict}. Confidence level: {confidence}."
                
                temp_dir = tempfile.gettempdir()
                temp_file = os.path.join(temp_dir, f"news_analysis_{int(time.time())}.mp3")
                
                tts = gTTS(text=speech_text, lang='en', slow=False)
                tts.save(temp_file)
                
                pygame.mixer.init()
                pygame.mixer.music.load(temp_file)
                pygame.mixer.music.play()
                
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                
                pygame.mixer.quit()
                time.sleep(0.5)
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                
                self.root.after(0, lambda: self.status_var.set("Audio summary delivered"))
                
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda: self.status_var.set(f"Audio error: {error_msg}"))
        
        threading.Thread(target=generate_speech, daemon=True).start()
    
    def fast_text_clean(self, text):
        text = str(text).lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def show_dataset_summary(self):
        """Display dataset summary"""
        if self.data is None:
            messagebox.showwarning("Warning", "Please initialize AI models first to load data")
            return
            
        self.print_to_console("\n" + "="*60)
        self.print_to_console("DATASET SUMMARY")
        self.print_to_console("="*60)
        
        self.print_to_console(f"Total Rows: {len(self.data):,}")
        self.print_to_console(f"Total Columns: {len(self.data.columns)}")
        self.print_to_console(f"Dataset Shape: {self.data.shape}")
        
        self.print_to_console("\nCOLUMNS AND DATA TYPES:")
        self.print_to_console("-" * 40)
        for col in self.data.columns:
            self.print_to_console(f"{col:20} : {str(self.data[col].dtype):15} | Unique: {self.data[col].nunique()}")
        
        self.print_to_console(f"\nMEMORY USAGE: {self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        self.print_to_console("\nLABEL DISTRIBUTION:")
        self.print_to_console("-" * 40)
        label_counts = self.data['label'].value_counts()
        self.print_to_console(f"Fake News (0): {label_counts.get(0, 0):,} samples ({label_counts.get(0, 0)/len(self.data)*100:.1f}%)")
        self.print_to_console(f"Real News (1): {label_counts.get(1, 0):,} samples ({label_counts.get(1, 0)/len(self.data)*100:.1f}%)")
        
        self.print_to_console("\nSAMPLE DATA (First 3 rows):")
        self.print_to_console("-" * 40)
        sample_data = self.data.head(3)[['title', 'text_length', 'word_count', 'label']].to_string()
        self.print_to_console(sample_data)
    
    def show_descriptive_stats(self):
        """Compute and display descriptive statistics"""
        if self.data is None:
            messagebox.showwarning("Warning", "Please initialize AI models first to load data")
            return
            
        self.print_to_console("\n" + "="*60)
        self.print_to_console("DESCRIPTIVE STATISTICS")
        self.print_to_console("="*60)
        
        numeric_cols = ['text_length', 'word_count', 'title_length']
        
        for col in numeric_cols:
            if col in self.data.columns:
                self.print_to_console(f"\nSTATISTICS FOR '{col.upper()}':")
                self.print_to_console("-" * 40)
                
                data = self.data[col]
                self.print_to_console(f"Mean: {data.mean():.2f}")
                self.print_to_console(f"Variance: {data.var():.2f}")
                self.print_to_console(f"Standard Deviation: {data.std():.2f}")
                self.print_to_console(f"Skewness: {data.skew():.4f}")
                self.print_to_console(f"Kurtosis: {data.kurtosis():.4f}")
                
                percentiles = [0, 25, 50, 75, 95, 100]
                self.print_to_console("\nPercentiles:")
                for p in percentiles:
                    self.print_to_console(f"  {p:2}th: {data.quantile(p/100):.2f}")
                
                self.print_to_console(f"\nRange: {data.min():.2f} - {data.max():.2f}")
                self.print_to_console(f"IQR: {data.quantile(0.75) - data.quantile(0.25):.2f}")
        
        # Additional statistics
        self.print_to_console(f"\nTEXT LENGTH BY LABEL:")
        self.print_to_console("-" * 40)
        fake_mean_len = self.data[self.data['label'] == 0]['text_length'].mean()
        true_mean_len = self.data[self.data['label'] == 1]['text_length'].mean()
        self.print_to_console(f"Fake News Avg Length: {fake_mean_len:.2f} characters")
        self.print_to_console(f"Real News Avg Length: {true_mean_len:.2f} characters")
    
    def generate_visualizations(self):
        """Generate EDA visualizations"""
        if self.data is None:
            messagebox.showwarning("Warning", "Please initialize AI models first to load data")
            return
            
        try:
            # Create a new window for visualizations
            viz_window = tk.Toplevel(self.root)
            viz_window.title("Exploratory Data Analysis - Visualizations")
            viz_window.geometry("1000x800")
            
            # Create notebook for multiple plots
            viz_notebook = ttk.Notebook(viz_window)
            viz_notebook.pack(fill='both', expand=True, padx=10, pady=10)
            
            # Plot 1: Histogram
            hist_frame = ttk.Frame(viz_notebook)
            viz_notebook.add(hist_frame, text="Text Length Distribution")
            
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            self.data['text_length'].hist(bins=30, alpha=0.7, color='skyblue', edgecolor='black', ax=ax1)
            ax1.set_title('Distribution of Text Lengths', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Text Length (characters)')
            ax1.set_ylabel('Frequency')
            ax1.grid(True, alpha=0.3)
            
            canvas1 = FigureCanvasTkAgg(fig1, hist_frame)
            canvas1.draw()
            canvas1.get_tk_widget().pack(fill='both', expand=True)
            
            # Plot 2: Boxplot
            box_frame = ttk.Frame(viz_notebook)
            viz_notebook.add(box_frame, text="Word Count by News Type")
            
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            fake_data = self.data[self.data['label'] == 0]
            true_data = self.data[self.data['label'] == 1]
            
            boxplot_data = [fake_data['word_count'], true_data['word_count']]
            ax2.boxplot(boxplot_data, labels=['Fake News', 'Real News'])
            ax2.set_title('Word Count Distribution by News Type', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Word Count')
            ax2.grid(True, alpha=0.3)
            
            canvas2 = FigureCanvasTkAgg(fig2, box_frame)
            canvas2.draw()
            canvas2.get_tk_widget().pack(fill='both', expand=True)
            
            # Plot 3: Correlation Heatmap
            corr_frame = ttk.Frame(viz_notebook)
            viz_notebook.add(corr_frame, text="Feature Correlation")
            
            fig3, ax3 = plt.subplots(figsize=(8, 6))
            numeric_data = self.data[['text_length', 'word_count', 'title_length', 'label']]
            correlation_matrix = numeric_data.corr()
            
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax3)
            ax3.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
            
            canvas3 = FigureCanvasTkAgg(fig3, corr_frame)
            canvas3.draw()
            canvas3.get_tk_widget().pack(fill='both', expand=True)
            
            self.print_to_console("✅ Visualizations generated successfully!")
            
        except Exception as e:
            self.print_to_console(f"❌ Error generating visualizations: {str(e)}")
    
    def show_all_model_evaluations(self):
        """Show comprehensive evaluation for all three models"""
        if not hasattr(self, 'model_evaluations'):
            messagebox.showwarning("Warning", "Please initialize AI models first to get evaluation results")
            return
            
        self.print_to_console("\n" + "="*60, "models")
        self.print_to_console("COMPREHENSIVE MODEL EVALUATIONS", "models")
        self.print_to_console("="*60, "models")
        
        for model_name, metrics in self.model_evaluations.items():
            self.print_to_console(f"\n📊 {model_name.upper()} EVALUATION:", "models")
            self.print_to_console("-" * 50, "models")
            
            self.print_to_console(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)", "models")
            self.print_to_console(f"Precision: {metrics['precision']:.4f}", "models")
            self.print_to_console(f"Recall:    {metrics['recall']:.4f}", "models")
            self.print_to_console(f"F1-Score:  {metrics['f1']:.4f}", "models")
            
            # Show confusion matrix
            self.print_to_console(f"\nConfusion Matrix:", "models")
            cm = metrics['confusion_matrix']
            self.print_to_console(f"[[{cm[0,0]:4} {cm[0,1]:4}]", "models")
            self.print_to_console(f" [{cm[1,0]:4} {cm[1,1]:4}]]", "models")
            
            self.print_to_console(f"\nClassification Report:", "models")
            self.print_to_console(metrics['classification_report'], "models")
    
    def compare_models(self):
        """Compare performance of all three models"""
        if not hasattr(self, 'model_evaluations'):
            messagebox.showwarning("Warning", "Please initialize AI models first to get evaluation results")
            return
            
        self.print_to_console("\n" + "="*60, "models")
        self.print_to_console("MODEL COMPARISON SUMMARY", "models")
        self.print_to_console("="*60, "models")
        
        # Create comparison table
        self.print_to_console(f"\n{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}", "models")
        self.print_to_console("-" * 60, "models")
        
        for model_name, metrics in self.model_evaluations.items():
            self.print_to_console(f"{model_name:<20} {metrics['accuracy']:.4f}    {metrics['precision']:.4f}    {metrics['recall']:.4f}    {metrics['f1']:.4f}", "models")
        
        # Find best model
        best_model = max(self.model_evaluations.items(), key=lambda x: x[1]['accuracy'])
        self.print_to_console(f"\n🏆 BEST PERFORMING MODEL: {best_model[0]}", "models")
        self.print_to_console(f"   Accuracy: {best_model[1]['accuracy']:.4f} ({best_model[1]['accuracy']*100:.2f}%)", "models")

    def train_models(self):
        self.status_var.set("Initializing AI models... This may take a few moments")
        self.print_to_console("\n🚀 STARTING AI MODEL INITIALIZATION...")
        
        def train_thread():
            try:
                # Step 1: Loading datasets
                self.root.after(0, lambda: self.status_var.set("Step 1/8: Loading datasets..."))
                fake_data = pd.read_csv("C:/Users/vansh/Downloads/Fake.csv")
                true_data = pd.read_csv("C:/Users/vansh/Downloads/True.csv")
                
                # REMOVE UNNAMED COLUMNS - METHOD 2
                fake_data = fake_data.loc[:, ~fake_data.columns.str.contains('^Unnamed')]
                true_data = true_data.loc[:, ~true_data.columns.str.contains('^Unnamed')]
                
                fake_data['label'] = 0
                true_data['label'] = 1
                
                # Use smaller dataset for testing
                fake_data = fake_data.sample(n=1000, random_state=42) if len(fake_data) > 1000 else fake_data
                true_data = true_data.sample(n=1000, random_state=42) if len(true_data) > 1000 else true_data
                
                self.data = pd.concat([fake_data, true_data], ignore_index=True)
                self.data = self.data.sample(frac=1, random_state=42).reset_index(drop=True)
                time.sleep(1)
                
                # Step 2: Feature engineering for EDA
                self.root.after(0, lambda: self.status_var.set("Step 2/8: Feature engineering..."))
                self.data['text_length'] = self.data['text'].apply(len)
                self.data['word_count'] = self.data['text'].apply(lambda x: len(str(x).split()))
                self.data['title_length'] = self.data['title'].apply(lambda x: len(str(x)))
                time.sleep(1)
                
                # Step 3: Handle missing values and outliers
                self.root.after(0, lambda: self.status_var.set("Step 3/8: Data preprocessing..."))
                self.data = self.data.fillna('')
                
                # Handle outliers in text_length
                Q1 = self.data['text_length'].quantile(0.25)
                Q3 = self.data['text_length'].quantile(0.75)
                IQR = Q3 - Q1
                upper_bound = Q3 + 1.5 * IQR
                self.data['text_length'] = np.where(self.data['text_length'] > upper_bound, upper_bound, self.data['text_length'])
                time.sleep(1)
                
                # Step 4: Text cleaning
                self.root.after(0, lambda: self.status_var.set("Step 4/8: Cleaning text data..."))
                self.data['cleaned_text'] = self.data['text'].apply(self.fast_text_clean)
                time.sleep(1)
                
                # Step 5: Feature extraction
                self.root.after(0, lambda: self.status_var.set("Step 5/8: Extracting features..."))
                self.vectorizer = TfidfVectorizer(
                    max_features=2000,
                    ngram_range=(1, 2),
                    stop_words='english',
                    min_df=2
                )
                
                X = self.vectorizer.fit_transform(self.data['cleaned_text'])
                y = self.data['label']
                time.sleep(1)
                
                # Step 6: Splitting data
                self.root.after(0, lambda: self.status_var.set("Step 6/8: Splitting data..."))
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                self.X_test, self.y_test = X_test, y_test
                time.sleep(1)
                
                # Step 7: Training models
                self.root.after(0, lambda: self.status_var.set("Step 7/8: Training models..."))
                
                # Train all three models
                lr_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
                lr_model.fit(X_train, y_train)
                
                rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
                rf_model.fit(X_train, y_train)
                
                nb_model = MultinomialNB()
                nb_model.fit(X_train, y_train)
                
                # Store all models
                self.models = {
                    'logistic_regression': lr_model,
                    'random_forest': rf_model,
                    'naive_bayes': nb_model
                }
                
                # Use Logistic Regression as the main model
                self.ensemble_model = lr_model
                time.sleep(1)
                
                # Step 8: Comprehensive evaluation
                self.root.after(0, lambda: self.status_var.set("Step 8/8: Evaluating models..."))
                self.model_evaluations = {}
                
                for model_name, model in self.models.items():
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)
                    
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    cm = confusion_matrix(y_test, y_pred)
                    report = classification_report(y_test, y_pred, target_names=['Fake News', 'Real News'])
                    
                    self.model_evaluations[model_name] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'confusion_matrix': cm,
                        'classification_report': report
                    }
                
                # Get overall accuracy
                overall_accuracy = self.model_evaluations['logistic_regression']['accuracy']
                
                self.root.after(0, lambda: self.show_training_result(overall_accuracy))
                self.print_to_console("🎉 AI MODEL INITIALIZATION COMPLETED SUCCESSFULLY!")
                
            except Exception as e:
                error_message = str(e)
                self.print_to_console(f"❌ INITIALIZATION FAILED: {error_message}")
                self.root.after(0, lambda: self.show_error(error_message))
        
        threading.Thread(target=train_thread, daemon=True).start()
    
    def show_training_result(self, accuracy):
        self.status_var.set(f"AI models initialized successfully - System Accuracy: {accuracy:.2%}")
        messagebox.showinfo("System Ready", 
                          f"AI Models Initialized Successfully!\n\n"
                          f"System Accuracy: {accuracy:.2%}\n\n"
                          f"Three models trained:\n"
                          f"• Logistic Regression\n"
                          f"• Random Forest\n"
                          f"• Naive Bayes\n\n"
                          f"Check the 'Model Evaluation' tab for detailed metrics.")
    
    def show_error(self, error_msg):
        self.status_var.set("Model initialization failed")
        messagebox.showerror("Initialization Error", f"Model initialization failed: {error_msg}")
    
    def analyze_text(self):
        text = self.text_input.get('1.0', tk.END).strip()
        if not text:
            messagebox.showwarning("Input Required", "Please enter news content or use voice input")
            return
            
        if not self.ensemble_model:
            messagebox.showwarning("System Not Ready", "Please initialize AI models first")
            return
            
        self.status_var.set("Analyzing content with AI models...")
        
        try:
            cleaned_text = self.fast_text_clean(text)
            X = self.vectorizer.transform([cleaned_text])
            prediction = self.ensemble_model.predict(X)[0]
            probabilities = self.ensemble_model.predict_proba(X)[0]
            
            fake_prob = probabilities[0]
            true_prob = probabilities[1]
            
            self.display_results(prediction, fake_prob, true_prob, text)
            self.update_confidence_display(max(fake_prob, true_prob))
            self.status_var.set("Analysis complete - Results ready")
            
        except Exception as e:
            error_message = str(e)
            messagebox.showerror("Analysis Error", f"Content analysis failed: {error_message}")
            self.status_var.set("Analysis failed")
    
    def display_results(self, prediction, fake_prob, true_prob, original_text):
        output = ""
        result = "FAKE NEWS 🚨" if prediction == 0 else "REAL NEWS ✅"
        confidence = max(fake_prob, true_prob)
        
        output += "FAKE NEWS DETECTION ANALYSIS REPORT\n\n\n"
        
        verdict_color = "🔴" if prediction == 0 else "🟢"
        output += f"{verdict_color} FINAL VERDICT: {result}\n\n"
        
        confidence_stars = "★" * int(confidence * 5) + "☆" * (5 - int(confidence * 5))
        confidence_color = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.5 else "🔴"
        output += f"{confidence_color} CONFIDENCE LEVEL: {confidence:.2%} {confidence_stars}\n\n"
        
        output += "📊 PROBABILITY BREAKDOWN:\n"
        fake_bar = "█" * int(fake_prob * 30) + "░" * (30 - int(fake_prob * 30))
        true_bar = "█" * int(true_prob * 30) + "░" * (30 - int(true_prob * 30))
        
        output += f"🔴 Fake News: {fake_prob:>6.2%} {fake_bar}\n\n"
        output += f"🟢 Real News: {true_prob:>6.2%} {true_bar}\n\n\n"
        
        output += "⚠️  RISK ASSESSMENT:\n"
        if fake_prob > 0.7:
            output += "🔴 HIGH RISK - Strong indicators of fake news detected\n"
            output += "   Recommendation: Verify through trusted sources\n"
        elif fake_prob > 0.6:
            output += "🟡 MEDIUM RISK - Suspicious content detected\n"
            output += "   Recommendation: Cross-check with fact-checkers\n"
        elif fake_prob > 0.4:
            output += "🟠 LOW RISK - Mixed signals, exercise caution\n"
            output += "   Recommendation: Additional verification needed\n"
        else:
            output += "🟢 LOW RISK - Content appears credible\n"
            output += "   Recommendation: Standard credibility checks\n"
        
        output += "\n"
        
        output += "📝 CONTENT PREVIEW:\n"
        preview = original_text[:200] + "..." if len(original_text) > 200 else original_text
        output += f"{preview}\n\n"
        
        output += "🔍 ANALYSIS DETAILS:\n"
        output += f"• Model Used: Logistic Regression\n"
        output += f"• Analysis Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        output += f"• Text Length: {len(original_text)} characters\n"
        output += f"• Decision Threshold: >60% confidence required\n"
        
        self.results_text.delete('1.0', tk.END)
        self.results_text.insert('1.0', output)
    
    def clear_all(self):
        self.text_input.delete('1.0', tk.END)
        self.results_text.delete('1.0', tk.END)
        self.eda_text.delete('1.0', tk.END)
        self.models_text.delete('1.0', tk.END)
        for widget in self.confidence_frame.winfo_children():
            widget.destroy()
        self.status_var.set("System Ready - All fields cleared")
        
    def run(self):
        self.print_to_console("🎯 FAKE NEWS DETECTION SYSTEM STARTED")
        self.print_to_console("👉 Click 'Initialize AI Models' to begin setup")
        self.root.mainloop()

if __name__ == "__main__":
    app = ProfessionalNewsDetector()
    app.run()
