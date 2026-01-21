import pandas as pd

def process_data():
    input_file = 'data/bitext_raw.csv'
    output_file = 'data/dataset.csv'
    
    print("Reading raw data...")
    df = pd.read_csv(input_file)
    
    # Mapping
    intent_map = {
        # Billing
        'check_invoice': 'Billing',
        'check_payment_methods': 'Billing',
        'check_refund_policy': 'Billing',
        'get_invoice': 'Billing',
        'get_refund': 'Billing',
        'payment_issue': 'Billing',
        'check_cancellation_fee': 'Billing',
        'track_refund': 'Billing',
        
        # Technical (Account & System issues)
        'create_account': 'Technical',
        'delete_account': 'Technical',
        'edit_account': 'Technical',
        'recover_password': 'Technical',
        'registration_problems': 'Technical',
        'switch_account': 'Technical',
        'newsletter_subscription': 'Technical', # Managing subscription settings often technical/account
        
        # General (Orders, Shipping, Contact, Feedback)
        'contact_customer_service': 'General',
        'contact_human_agent': 'General',
        'complaint': 'General',
        'review': 'General',
        'delivery_options': 'General',
        'delivery_period': 'General',
        'track_order': 'General',
        'place_order': 'General',
        'change_order': 'General',
        'cancel_order': 'General',
        'change_shipping_address': 'General',
        'set_up_shipping_address': 'General'
    }
    
    # Apply mapping
    print("Mapping intents to categories...")
    df['category'] = df['intent'].map(intent_map)
    
    # Check for unmapped
    unmapped = df[df['category'].isna()]['intent'].unique()
    if len(unmapped) > 0:
        print(f"Warning: Unmapped intents found: {unmapped}")
        # Default to General
        df['category'] = df['category'].fillna('General')
    
    # Select columns
    # instruction is the user text
    final_df = df[['instruction', 'category']].rename(columns={'instruction': 'text'})
    
    # Add synthetic data for Pricing/Plans -> Billing
    # The Bitext dataset is low on explicit "plan comparison" queries, so we inject some to help the model learn.
    print("Injecting synthetic pricing/plan data...")
    pricing_data = [
        {'text': 'What is the difference between basic and pro plans?', 'category': 'Billing'},
        {'text': 'How much does the premium plan cost?', 'category': 'Billing'},
        {'text': 'Tell me about your pricing tiers.', 'category': 'Billing'},
        {'text': 'Can you help me understand the difference between the basic and pro plans?', 'category': 'Billing'},
        {'text': 'Is there a free trial for the pro plan?', 'category': 'Billing'},
        {'text': 'Upgrade to enterprise plan cost', 'category': 'Billing'},
        {'text': 'Compare subscription plans', 'category': 'Billing'},
        {'text': 'What features are included in the basic plan?', 'category': 'Billing'},
        {'text': 'Pricing for small businesses', 'category': 'Billing'},
        {'text': 'Do you offer monthly or yearly billing for the pro plan?', 'category': 'Billing'},
        {'text': 'I want to upgrade my subscription to the next tier', 'category': 'Billing'},
        {'text': 'What are the benefits of the gold plan?', 'category': 'Billing'}
    ]
    
    # Duplicate these samples to ensure they have enough weight (27k total samples, so ~500 copies gives ~0.5% weight which is significant enough for these specific patterns)
    pricing_df = pd.DataFrame(pricing_data)
    pricing_df_augmented = pd.concat([pricing_df] * 50, ignore_index=True)
    
    final_df = pd.concat([final_df, pricing_df_augmented], ignore_index=True)

    # Add synthetic data for Access/Feature Locking Issues -> Technical
    # User feedback: "After the update, premium features are locked even though payment was successful" should be Technical.
    # The model currently confuses this with Billing because of the word "payment".
    print("Injecting synthetic access/feature issue data...")
    access_data = [
        {'text': 'After the update, premium features are locked even though payment was successful.', 'category': 'Technical'},
        {'text': 'I paid for the subscription but cannot access pro features.', 'category': 'Technical'},
        {'text': 'Premium content is locked despite active subscription.', 'category': 'Technical'},
        {'text': 'My account shows free plan but I have already paid.', 'category': 'Technical'},
        {'text': 'Features not unlocking after upgrade.', 'category': 'Technical'},
        {'text': 'I cannot access the dashboard after payment.', 'category': 'Technical'},
        {'text': 'License key not working after purchase.', 'category': 'Technical'},
        {'text': 'App says upgrade required but I am already a premium member.', 'category': 'Technical'},
        {'text': 'Cannot access paid tools after update.', 'category': 'Technical'},
        {'text': 'Subscription is active but features are disabled.', 'category': 'Technical'}
    ]
    
    access_df = pd.DataFrame(access_data)
    # Higher weight to override strong "payment" -> "Billing" association
    access_df_augmented = pd.concat([access_df] * 100, ignore_index=True)
    
    final_df = pd.concat([final_df, access_df_augmented], ignore_index=True)

    # Add synthetic data for Technical Failures in Billing Context -> Technical
    # User feedback: "The app freezes whenever I try to download my billing statement" should be Technical.
    # The model currently confuses this with Billing because of "billing statement".
    print("Injecting synthetic technical faults in billing context...")
    tech_billing_data = [
        {'text': 'The app freezes whenever I try to download my billing statement.', 'category': 'Technical'},
        {'text': 'I get an error 404 when clicking on invoice.', 'category': 'Technical'},
        {'text': 'The billing page crashes when I load it.', 'category': 'Technical'},
        {'text': 'Cannot download receipt, button is not working.', 'category': 'Technical'},
        {'text': 'System times out when processing payment.', 'category': 'Technical'},
        {'text': 'Blank screen when viewing payment history.', 'category': 'Technical'},
        {'text': 'App crashes when I try to update credit card.', 'category': 'Technical'},
        {'text': 'Download PDF failed for my latest invoice.', 'category': 'Technical'},
        {'text': 'Website is down so I cannot pay my bill.', 'category': 'Technical'},
        {'text': 'Login failed on the billing portal.', 'category': 'Technical'}
    ]
    
    tech_billing_df = pd.DataFrame(tech_billing_data)
    # Heavy weighting to overcome strong keyword associations
    tech_billing_df_augmented = pd.concat([tech_billing_df] * 100, ignore_index=True)
    
    final_df = pd.concat([final_df, tech_billing_df_augmented], ignore_index=True)
    
    # Save
    final_df.to_csv(output_file, index=False)
    print(f"Processed {len(final_df)} samples.")
    print(f"Saved to {output_file}")
    print("\nClass distribution:")
    print(final_df['category'].value_counts())
    print("\nSample data:")
    print(final_df.head())

if __name__ == "__main__":
    process_data()
