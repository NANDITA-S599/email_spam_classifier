#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "spam_classifier.h"

void display_menu() {
    printf("\n=== Email Spam Classifier ===\n");
    printf("1. Train with default data\n");
    printf("2. Train from file\n");
    printf("3. Classify email\n");
    printf("4. Save model\n");
    printf("5. Load model\n");
    printf("6. Interactive classification\n");
    printf("7. Test accuracy\n");
    printf("8. Exit\n");
    printf("Choose an option: ");
}

void train_default_data(Vocabulary* vocab) {
    printf("\nTraining with default dataset...\n");
    
    // Spam examples
    train_classifier(vocab, "win free money now click here urgent", 1);
    train_classifier(vocab, "congratulations you won lottery claim prize", 1);
    train_classifier(vocab, "urgent your account has been compromised", 1);
    train_classifier(vocab, "limited time offer buy now discount", 1);
    train_classifier(vocab, "inheritance money transfer fees required", 1);
    train_classifier(vocab, "you are selected for special promotion", 1);
    train_classifier(vocab, "claim your free gift now limited", 1);
    
    // Ham examples
    train_classifier(vocab, "meeting scheduled for tomorrow please attend", 0);
    train_classifier(vocab, "project deadline next week team collaboration", 0);
    train_classifier(vocab, "lunch together friday restaurant reservation", 0);
    train_classifier(vocab, "family dinner this weekend bring dessert", 0);
    train_classifier(vocab, "software update available security patch install", 0);
    train_classifier(vocab, "weekly report attached please review", 0);
    train_classifier(vocab, "birthday party next month save the date", 0);
    
    printf("Default training completed!\n");
    printf("Spam emails: %d, Ham emails: %d\n", 
           vocab->total_spam_emails, vocab->total_ham_emails);
    printf("Vocabulary size: %d words\n", get_vocabulary_size(vocab));
}

void classify_interactive(Vocabulary* vocab) {
    char email[MAX_EMAIL_LENGTH];
    
    printf("\nEnter email to classify (or 'quit' to exit):\n");
    printf("> ");
    
    while (fgets(email, sizeof(email), stdin)) {
        // Remove newline
        email[strcspn(email, "\n")] = 0;
        
        if (strcmp(email, "quit") == 0) break;
        if (strlen(email) == 0) continue;
        
        double probability = calculate_probability(vocab, email);
        printf("Spam probability: %.2f%%\n", probability * 100);
        
        if (probability > 0.7) {
            printf("Classification: SPAM \n");
        } else if (probability < 0.3) {
            printf("Classification: HAM \n");
        } else {
            printf("Classification: UNCERTAIN ⚠\n");
        }
        
        printf("\n> ");
    }
}

void test_accuracy(Vocabulary* vocab) {
    printf("\n=== Testing Classifier Accuracy ===\n");
    
    // Test cases
    struct TestCase {
        char* email;
        int expected_class;
        char* description;
    };
    
    struct TestCase tests[] = {
        {"win free money now click here", 1, "Obvious spam"},
        {"meeting tomorrow at conference room", 0, "Clear ham"},
        {"congratulations you won prize", 1, "Spam with winning"},
        {"lunch meeting with team today", 0, "Work email"},
        {"urgent account verification required", 1, "Phishing attempt"},
        {"project deadline extended to friday", 0, "Project update"},
        {"claim your free gift now", 1, "Spam with free"},
        {"family dinner this weekend", 0, "Personal email"}
    };
    
    int total_tests = sizeof(tests) / sizeof(tests[0]);
    int correct = 0;
    
    for (int i = 0; i < total_tests; i++) {
        double prob = calculate_probability(vocab, tests[i].email);
        int predicted = (prob > 0.5) ? 1 : 0;
        
        printf("\nTest %d: %s\n", i+1, tests[i].description);
        printf("Email: %s\n", tests[i].email);
        printf("Expected: %s, Predicted: %s, Probability: %.2f%%\n",
               tests[i].expected_class ? "SPAM" : "HAM",
               predicted ? "SPAM" : "HAM",
               prob * 100);
        
        if (predicted == tests[i].expected_class) {
            printf("✅ CORRECT\n");
            correct++;
        } else {
            printf("❌ WRONG\n");
        }
    }
    
    printf("\n=== Results ===\n");
    printf("Accuracy: %d/%d (%.2f%%)\n", correct, total_tests, 
           (double)correct/total_tests * 100);
}

int main() {
    Vocabulary vocab;
    init_vocabulary(&vocab);
    
    int choice;
    char filename[100];
    char email[MAX_EMAIL_LENGTH];
    
    printf("Email Spam Classifier in C\n");
    printf("==========================\n");
    printf("Naive Bayes Implementation with Hash Tables\n\n");
    
    while (1) {
        display_menu();
        if (scanf("%d", &choice) != 1) {
            printf("Invalid input!\n");
            while (getchar() != '\n'); // Clear input buffer
            continue;
        }
        getchar(); // Consume newline
        
        switch (choice) {
            case 1:
                train_default_data(&vocab);
                break;
                
            case 2:
                printf("Enter training file name: ");
                fgets(filename, sizeof(filename), stdin);
                filename[strcspn(filename, "\n")] = 0;
                load_training_data(&vocab, filename);
                break;
                
            case 3:
                if (vocab.total_spam_emails == 0 || vocab.total_ham_emails == 0) {
                    printf("Error: No training data available! Train first.\n");
                } else {
                    printf("Enter email to classify: ");
                    fgets(email, sizeof(email), stdin);
                    email[strcspn(email, "\n")] = 0;
                    
                    double probability = calculate_probability(&vocab, email);
                    printf("Spam probability: %.2f%%\n", probability * 100);
                    
                    if (probability > 0.7) {
                        printf("Classification: SPAM 🚨\n");
                    } else if (probability < 0.3) {
                        printf("Classification: HAM ✅\n");
                    } else {
                        printf("Classification: UNCERTAIN ⚠\n");
                    }
                }
                break;
                
            case 4:
                printf("Enter filename to save model: ");
                fgets(filename, sizeof(filename), stdin);
                filename[strcspn(filename, "\n")] = 0;
                save_model(&vocab, filename);
                break;
                
            case 5:
                printf("Enter filename to load model: ");
                fgets(filename, sizeof(filename), stdin);
                filename[strcspn(filename, "\n")] = 0;
                load_model(&vocab, filename);
                break;
                
            case 6:
                if (vocab.total_spam_emails == 0 || vocab.total_ham_emails == 0) {
                    printf("Error: No training data available! Train first.\n");
                } else {
                    classify_interactive(&vocab);
                }
                break;
                
            case 7:
                if (vocab.total_spam_emails == 0 || vocab.total_ham_emails == 0) {
                    printf("Error: No training data available! Train first.\n");
                } else {
                    test_accuracy(&vocab);
                }
                break;
                
            case 8:
                free_vocabulary(&vocab);
                printf("Goodbye!\n");
                exit(0);
                
            default:
                printf("Invalid choice! Please try again.\n");
        }
    }
    
    return 0;
}