import streamlit as st
import requests
import pandas as pd
try:
    import plotly.express as px
    import plotly.figure_factory as ff
except ImportError:
    st.error("Plotly is not installed. Please install it with 'pip install plotly' to enable charts.")
    px = None
    ff = None

# Configuration
API_URL = "http://localhost:8000/predict"

def main():
    st.set_page_config(
        page_title="Business Text Classifier",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 Business Text Classification System")
    
    # Create tabs
    tab1, tab2 = st.tabs(["Prediction", "Model Evaluation"])
    
    with tab1:
        st.markdown("""
        Classify business communications into **Technical**, **Billing**, or **General** categories.
        """)

    # Sidebar for info
    with st.sidebar:
        st.header("About")
        st.info("This system uses Machine Learning models to categorize text inputs.")
        
        # Model Selection
        st.subheader("Model Selection")
        model_choice = st.radio(
            "Choose a model:",
            ("BERT (Transformer)", "LinearSVC (Scikit-Learn)")
        )
        model_key = "bert" if "BERT" in model_choice else "sklearn"

        st.markdown("### Categories:")
        st.markdown("- **Technical**: Server issues, bugs, crashes")
        st.markdown("- **Billing**: Invoices, payments, refunds")
        st.markdown("- **General**: General inquiries, greetings")

    # Main input area
    col1, col2 = tab1.columns([1, 1])

    with col1:
        st.subheader("Input Text")
        text_input = st.text_area(
            "Enter the text you want to classify:",
            height=200,
            placeholder="e.g., My server is down and I cannot login..."
        )
        
        if st.button("Classify Text", type="primary"):
            if text_input.strip():
                try:
                    with st.spinner(f"Analyzing with {model_choice}..."):
                        response = requests.post(API_URL, json={"text": text_input, "model": model_key})
                        
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display results in the second column
                        with col2:
                            st.subheader("Analysis Results")
                            
                            # Top prediction
                            category = result['category']
                            confidence = result['confidence']
                            
                            # Color coding based on category
                            color_map = {
                                "Technical": "red",
                                "Billing": "green",
                                "General": "blue"
                            }
                            color = color_map.get(category, "gray")
                            
                            st.markdown(f"### Predicted Category: :{color}[{category}]")
                            st.metric("Confidence Score", f"{confidence:.2%}")
                            
                            # Probability chart
                            st.markdown("#### Probability Distribution")
                            probs = result.get('probabilities', {})
                            if probs:
                                df_probs = pd.DataFrame(list(probs.items()), columns=['Category', 'Probability'])
                                fig = px.bar(
                                    df_probs, 
                                    x='Category', 
                                    y='Probability',
                                    color='Category',
                                    range_y=[0, 1],
                                    color_discrete_map=color_map
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the API. Is the server running?")
            else:
                st.warning("Please enter some text to classify.")

    # Example inputs
    st.divider()
    st.subheader("Try Examples")
    examples = [
        "My internet connection is very slow today.",
        "I was charged twice for my subscription.",
        "Where is your office located?",
        "The API returns a 500 error when I submit the form."
    ]
    
    for ex in examples:
        if st.button(ex):
            # We can't easily populate the text area programmatically without session state tricks
            # So we'll just run the classification directly for the example
            try:
                response = requests.post(API_URL, json={"text": ex, "model": model_key})
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"**Category:** {result['category']} (Confidence: {result['confidence']:.2%})")
            except:
                st.error("API Error")

    with tab2:
        st.header("Model Evaluation Metrics")
        
        import json
        import os
        
        # Load metrics
        metrics_file = "models/metrics_bert.json" if model_key == "bert" else "models/metrics_linear.json"
        
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            # Display high-level metrics
            m1, m2 = st.columns(2)
            m1.metric("Test Accuracy", f"{metrics['accuracy']:.2%}")
            m2.metric("F1 Score (Weighted)", f"{metrics['f1_score']:.4f}")
            
            st.divider()
            
            # Display detailed classification report
            st.subheader("Detailed Classification Report")
            report = metrics.get('classification_report', {})
            if report:
                # Convert to dataframe for nicer display
                df_report = pd.DataFrame(report).transpose()
                # Filter out accuracy/macro avg rows if desired, or keep them
                st.dataframe(df_report.style.format("{:.4f}"))
            
            st.divider()
            
            # Display Confusion Matrix
            st.subheader("Confusion Matrix")
            cm = metrics.get('confusion_matrix', [])
            if cm and ff:
                # Labels
                labels = ['Billing', 'General', 'Technical']
                
                # Invert for heatmap (y-axis top to bottom) usually handled by library
                fig_cm = ff.create_annotated_heatmap(
                    z=cm, 
                    x=labels, 
                    y=labels, 
                    colorscale='Blues',
                    showscale=True
                )
                fig_cm.update_layout(title_text='Confusion Matrix', xaxis_title="Predicted", yaxis_title="True Label")
                st.plotly_chart(fig_cm, use_container_width=True)
                
        else:
            st.warning(f"Metrics file not found for {model_choice}. Please train the model first.")

if __name__ == "__main__":
    main()
