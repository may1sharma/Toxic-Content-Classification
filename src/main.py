import sys
import model

def create_model():
    model.create_and_save()

def default_main():
    model.predict_score()

def main(comment):
    model.predict_individual_score(comment)

if __name__ == '__main__':
    if (len(sys.argv) == 2):
        if(sys.argv[1] == 'create'):
            create_model()
        else:
            main(sys.argv[1])
    elif (len(sys.argv) == 1):
        default_main()
    else:
        print ('Usage:\tmain.py <optional: Comment / Create Model["create"]>')
        sys.exit(0)