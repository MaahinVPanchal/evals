import evals
import evals.metrics
import random
import openai
import logging
import json

# Configure the logging
log_file_path = 'D:/Maahin_Coding/ML_Project/logs/evaluation.log'  # Specify your log file path
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,  # Change to logging.DEBUG for detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ExtractCountries(evals.Eval):
    def __init__(self, test_jsonl, **kwargs):  # ✅ Fixed __init__ method
        super().__init__(**kwargs)
        self.test_jsonl = test_jsonl
        logging.info(f"Initialized ExtractCountries eval with test file: {self.test_jsonl}")

        # ✅ Load dataset
        try:
            with open(self.test_jsonl, "r", encoding="utf-8") as file:
                self.test_samples = [json.loads(line) for line in file]
        except Exception as e:
            logging.error(f"Error loading test JSONL file: {e}")
            self.test_samples = []

    def run(self, recorder):
        if not self.test_samples:
            logging.error("No test samples found. Exiting evaluation.")
            return {"accuracy": 0.0}

        logging.info(f"Running evaluation on dataset: {self.test_jsonl}")
        self.eval_all_samples(recorder, self.test_samples)
        
        accuracy = evals.metrics.get_accuracy(recorder.get_events("match"))
        logging.info(f"Evaluation complete with accuracy: {accuracy}")

        return {"accuracy": accuracy}

    def eval_sample(self, test_sample, rng: random.Random):
        logging.info(f"Evaluating sample with input: {test_sample['input']}")
        prompt = test_sample["input"]
        result = self.completion_fn(
            prompt=prompt,
            max_tokens=25
        )
        
        # Fixing the error
        sampled = result.get_completions()[0] if result.get_completions() else ""

        evals.record_and_check_match(
            prompt,
            sampled,
            expected=test_sample["ideal"]
        )
        logging.info(f"Sample evaluated. Expected: {test_sample['ideal']}, Got: {sampled}")

