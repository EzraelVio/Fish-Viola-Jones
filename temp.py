# Set Initial Parameters
initial_fpr = 0.5  # Initial False Positive Rate
initial_dr = 0.99  # Initial Detection Rate
min_acceptable_dr = 0.95  # Minimum Acceptable Detection Rate for Each Stage

# Initialize Cascade
cascade = []

# Define Function to Calculate False Positive Rate (FPR) and Detection Rate (DR)
def evaluate_cascade(cascade, data, labels):
    total_samples = len(data)
    false_positives = 0
    detections = 0
    
    for i in range(total_samples):
        sample = data[i]
        label = labels[i]
        
        # Check each stage in the cascade
        for stage_classifiers in cascade:
            stage_decision = "positive"  # Initialize as positive
            
            # Evaluate each classifier in the stage
            for classifier, alpha in stage_classifiers:
                # Replace this line with the actual prediction using your classifier
                prediction = classifier.predict([sample])[0]
                
                # Check if the classifier rejects the sample
                if prediction == 0:  # Replace 0 with the label representing rejection
                    stage_decision = "negative"
                    break
            
            # If any classifier in the stage rejects, break to the next stage
            if stage_decision == "negative":
                break
            else:
                # Increment detection count if all classifiers in the stage accept
                detections += 1
        
        # Increment false positives if not all stages accept
        if stage_decision == "negative":
            false_positives += 1
    
    # Calculate FPR and DR
    fpr = false_positives / total_samples
    dr = detections / total_samples
    
    return fpr, dr

# Initialize Cascade
current_dr = 1.0
current_fpr = 0.0
current_stage = []

# Iterate through sorted classifiers
for classifier, alpha in sorted_classifiers:
    # Add classifier to current stage
    current_stage.append((classifier, alpha))
    
    # Evaluate current cascade
    current_fpr, current_dr = evaluate_cascade([current_stage], X_valid, Y_valid)
    
    # Check if current FPR is below the target
    if current_fpr <= initial_fpr:
        # Check if current DR is above the target or meets the minimum acceptable DR
        if current_dr >= initial_dr or current_dr >= min_acceptable_dr:
            # Add current stage to the cascade
            cascade.append(current_stage)
            print(f"Stage added with FPR: {current_fpr}, DR: {current_dr}")
            
            # Reset current stage for the next iteration
            current_stage = []

# Print the final cascade
print("Final Cascade:")
for i, stage_classifiers in enumerate(cascade):
    print(f"Stage {i + 1}: {len(stage_classifiers)} classifiers")
