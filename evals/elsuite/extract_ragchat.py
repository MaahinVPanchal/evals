import json
import numpy as np
import redis
from django.http import JsonResponse
import logging
import evals
import evals.metrics
import random
import openai
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialize Redis connection
redis_client_embeddings = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)

# Set up logging
logger = logging.getLogger(__name__)
log_file_path = "D:/Maahin_Coding/ML_Project/logs/evaluation.log"
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

class Extractrag(evals.Eval):
    def __init__(self, test_jsonl, **kwargs):
        super().__init__(**kwargs)
        self.test_jsonl = test_jsonl
        logging.info(f"Initialized Extractrag eval with test file: {self.test_jsonl}")
        self.test_samples = self.load_test_samples()

    def load_test_samples(self):
        logging.info(f"Loading test file from: {self.test_jsonl}")
        try:
            with open(self.test_jsonl, "r", encoding="utf-8") as file:
                samples = [json.loads(line) for line in file]

            valid_samples = []
            for sample in samples:
                query = sample.get("query")
                embedding = sample.get("embedding")
                doc_id = sample.get("doc_id")

                if query is None or embedding is None or doc_id is None:
                    logging.warning(f"Skipping sample due to missing fields: {sample}")
                    continue

                if not isinstance(embedding, list):
                    logging.warning(f"Invalid embedding format in sample: {sample}")
                    continue

                top_k_texts = self.retrieve_top_k_similar(doc_id, embedding)
                print(f"Top 5 matched texts for query '{query}' with doc_id '{doc_id}': {top_k_texts}")
                sample["top_k_texts"] = top_k_texts
                sample["summary"] = self.summarize_texts( query,top_k_texts)
                print(f"Summarize '{query}' with doc_id '{doc_id}' and its summary is :{sample["summary"]}")
                valid_samples.append(sample)
                
            return valid_samples
        except FileNotFoundError:
            logging.error(f"Test JSONL file not found: {self.test_jsonl}")
        except json.JSONDecodeError:
            logging.error("Error decoding JSONL file. Ensure it is properly formatted.")
        except Exception as e:
            logging.error(f"Unexpected error loading test JSONL file: {e}")
        return []

    def run(self, recorder):
        if not self.test_samples:
            logging.error("No valid test samples found. Exiting evaluation.")
            return {"accuracy": 0.0}

        logging.info(f"Running evaluation on dataset: {self.test_jsonl}")
        self.eval_all_samples(recorder, self.test_samples)

        accuracy = evals.metrics.get_accuracy(recorder.get_events("match"))
        logging.info(f"Evaluation complete. Accuracy: {accuracy}")

        return {"accuracy": accuracy}



    def eval_sample(self, test_sample, rng: random.Random):
        query = test_sample.get("query", "")
        embedding = test_sample.get("embedding", [])
        expected = test_sample.get("summary", "")  # Get expected response (GPT-4)

        if not query or not expected:
            logging.warning("Skipping sample due to missing query or expected summary.")
            return

        logging.info(f"Evaluating sample with query: {query}")

        try:
            result = self.completion_fn(prompt=query, max_tokens=125)
            completions = result.get_completions()
            sampled = completions[0] if completions else ""

            # Compute similarity between expected and generated response
            similarity = self.compute_similarity(expected, sampled)

            # Set threshold (e.g., 0.8) to count as a correct match
            is_correct = similarity >= 0.8
            evals.record_and_check_match(query, sampled, expected=expected)

            logging.info(f"Expected: {expected}\nGot: {sampled}\nSimilarity: {similarity}")
            return is_correct  # Return True if the similarity is high enough
        except Exception as e:
            logging.error(f"Error in OpenAI completion: {e}")

    def compute_similarity(self, text1, text2):
        """Compute cosine similarity between two text responses using OpenAI embeddings."""
        try:
            client = openai.OpenAI()
            embeddings = client.embeddings.create(
                input=[text1, text2],
                model="text-embedding-ada-002"
            )

            vec1 = np.array(embeddings.data[0].embedding).reshape(1, -1)
            vec2 = np.array(embeddings.data[1].embedding).reshape(1, -1)

            similarity = cosine_similarity(vec1, vec2)[0][0]
            return similarity
        except Exception as e:
            logging.error(f"Error computing similarity: {e}")
            return 0.0

    def retrieve_top_k_similar(self, doc_id, query_embedding, top_k=10):
        try:
            key = f"embedding:{doc_id}"
            stored_data = redis_client_embeddings.hget(key, "embeddings")
            if not stored_data:
                logging.warning(f"No embeddings found in Redis for doc_id: {doc_id}.")
                return []

            stored_data = json.loads(stored_data)
            stored_embeddings = stored_data if isinstance(stored_data, list) else [stored_data]

            similarity_scores = []
            for entry in stored_embeddings:
                stored_embedding = entry.get("embedding", [])
                if not stored_embedding:
                    continue

                if len(query_embedding) != len(stored_embedding):
                    logging.error(f"Dimension mismatch: query({len(query_embedding)}) vs stored({len(stored_embedding)})")
                    continue

                similarity = np.dot(query_embedding, stored_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
                )

                similarity_scores.append((entry["text"], similarity))

            top_k_results = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:top_k]

            top_k_texts = [text for text, _ in top_k_results]
            print(f"Matched top {top_k} texts for doc_id {doc_id}: {top_k_texts}")
            logging.info(f"Top {top_k} similar texts for doc_id {doc_id}: {top_k_texts}")

            return top_k_texts
        except Exception as e:
            logging.error(f"Error retrieving top {top_k} similar embeddings for doc_id {doc_id}: {e}")
            return []

    import openai
    import logging

    def summarize_texts(self, query, texts):
        if not texts:
            return ""

        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": f"Answer the following texts based on the query: '{query}'"},
                    {"role": "user", "content": "\n".join(texts)}
                ]
            )

            # Ensure response is valid
            summary = response.choices[0].message.content.strip() if response.choices else ""
            
            if not summary:
                logging.warning("Answered returned an empty response.")
            
            return summary

        except Exception as e:
            logging.error(f"Error in answer: {e}")
            return ""

