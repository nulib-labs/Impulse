import os
from fireworks.utilities.filepad import FilePad
from pymongo import MongoClient
import json
from bs4 import BeautifulSoup
import os
from typing import Optional, List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from loguru import logger
from typing import List, Tuple
import pickle


class Kristeva:
    embedding_model: Optional[Any] = None
    embedding_batch_size: int = 32
    similarity_threshold: float = 0.8
    top_k: Optional[int] = None
    verbose = True

    def __init__(
        self,
        similarity_threshold: float | None = None,
        top_k: int | None = None,
        embedding_model: Any = None,
        verbose: bool = True,
    ):
        self._similarity_threshold: float = (
            similarity_threshold
            if similarity_threshold is not None
            else self.similarity_threshold
        )
        self.top_k = top_k
        self.embedding_model = embedding_model
        self._sentence_df: Optional[pd.DataFrame] = None
        self._verbose = verbose
        self._G: Optional[nx.Graph] = None
        self._sentence_embeddings: Optional[np.ndarray] = None
        self._similarity_matrix: Optional[np.ndarray] = None

    def _split_documents(self, documents: Dict[Any, List[str]]) -> pd.DataFrame:
        rows = []
        for doc_id, sentences in documents.items():
            for sent_pos, sent in enumerate(sentences):
                sent = str(sent).strip()
                if not sent:
                    continue
                rows.append(
                    {
                        "document_id": doc_id,
                        "sentence_position": sent_pos,
                        "sentence": sent,
                    }
                )
        return pd.DataFrame(rows)

    def train(
        self,
        documents: Dict[Any, List[str]],
        embeddings: Optional[np.ndarray] = None,
        verbose: Optional[bool] = None,
    ) -> nx.Graph:
        if verbose is None:
            verbose = self._verbose
        # Build sentence dataframe
        df = self._split_documents(documents)
        self._sentence_df = df

        # Embed sentences
        if embeddings is None:
            if self.embedding_model is None:
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            if verbose:
                self.embedding_model: SentenceTransformer
                print("Encoding sentences...")
                embeddings = self.embedding_model.encode(
                    df["sentence"].tolist(),
                    batch_size=self.embedding_batch_size,
                    show_progress_bar=True,
                )
            else:
                embeddings = self.embedding_model.encode(
                    df["sentence"].tolist(),
                    batch_size=self.embedding_batch_size,
                    show_progress_bar=False,
                )
        embeddings = np.asarray(embeddings)
        self._sentence_embeddings = embeddings  # Store embeddings
        df["sentence_embedding"] = list(embeddings)

        # Precompute similarity matrix across all sentences
        from sklearn.metrics.pairwise import cosine_similarity

        if verbose:
            logger.info("Now computing cosine similarity matrix.")
        sim_matrix = cosine_similarity(embeddings)
        self._similarity_matrix = sim_matrix  # Store similarity matrix

        # Index sentences per document
        doc_to_indices = df.groupby(
            "document_id"
        ).indices  # mapping doc_id -> list of row indices

        # Create graph at document level
        if verbose:
            logger.info("Building document graph...")
        G = nx.Graph()
        for doc_id, group in df.groupby("document_id"):
            G.add_node(
                doc_id,
                document_id=doc_id,
                sentences=group["sentence"].tolist(),
            )

        # Build edges: for each doc pair, check sentence pair similarities
        doc_ids = sorted(doc_to_indices.keys())
        for i in range(len(doc_ids)):
            for j in range(i + 1, len(doc_ids)):
                d1, d2 = doc_ids[i], doc_ids[j]
                idx1 = doc_to_indices[d1]
                idx2 = doc_to_indices[d2]

                # Extract submatrix similarities between sentences of doc1 and doc2
                sub = sim_matrix[np.ix_(idx1, idx2)]

                # Find max similarity exceeding threshold
                max_sim = np.max(sub) if sub.size else 0.0
                if max_sim > self._similarity_threshold:
                    # Optionally capture which sentences matched (only those above threshold)
                    match_positions: List[Tuple[int, int, float]] = []
                    rows, cols = np.where(sub > self._similarity_threshold)
                    for r, c in zip(rows, cols):
                        match_positions.append(
                            (
                                int(df.iloc[idx1[r]]["sentence_position"]),
                                int(df.iloc[idx2[c]]["sentence_position"]),
                                float(sub[r, c]),
                            )
                        )
                    G.add_edge(
                        d1,
                        d2,
                        weight=float(max_sim),
                        max_sentence_similarity=float(max_sim),
                        matched_sentence_pairs=match_positions,
                    )

        # After building all edges, before community detection
        if self.top_k is not None:
            for node in list(G.nodes()):
                edges = list(G.edges(node, data=True))
                if len(edges) > self.top_k:
                    edges.sort(key=lambda e: e[2].get("weight", 0), reverse=True)
                    to_remove = edges[self.top_k :]
                    for u, v, _ in to_remove:
                        if G.has_edge(u, v):
                            G.remove_edge(u, v)

        # Community detection (optional)
        from networkx.algorithms.community import greedy_modularity_communities

        if G.number_of_edges() > 0:
            for cid, comm in enumerate(greedy_modularity_communities(G)):
                for node in comm:
                    G.nodes[node]["community"] = cid
        else:
            for node in G.nodes:
                G.nodes[node]["community"] = 0

        self._G = G
        return G

    def visualize_graph(self, G: nx.Graph) -> go.Figure:
        pos = nx.spring_layout(G, seed=42, weight="weight")
        edge_x, edge_y, edge_text = [], [], []
        for u, v, data in G.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
            edge_text.append(
                f"{u} ↔ {v} max_sim={data.get('max_sentence_similarity'):.3f}"
            )

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1.0, color="#888"),
            hoverinfo="none",
            mode="lines",
        )

        node_x, node_y, node_text, node_color = [], [], [], []
        for node, data in G.nodes(data=True):
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_color.append(data.get("community", 0))
            node_text.append(
                f"Doc {node}<br>"
                + "<br>".join(data.get("sentences", [])[:5])
                + ("<br>..." if len(data.get("sentences", [])) > 5 else "")
            )

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            hoverinfo="text",
            marker=dict(
                showscale=True,
                colorscale="Viridis",
                color=node_color,
                size=22,
                line_width=1,
                colorbar=dict(title="Community", thickness=15),
            ),
            text=node_text,
        )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title="<br>Document Graph (Edges via sentence similarity)",
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            ),
        )
        return fig

    def save_model(self, filepath: str):
        """Saves the trained model, including sentence embeddings,
        similarity matrix, and sentence dataframe."""
        if (
            self._sentence_embeddings is None
            or self._similarity_matrix is None
            or self._sentence_df is None
        ):
            raise ValueError("Model must be trained before saving.")

        # Create a directory if it doesn't exist
        dir_path = os.path.dirname(os.path.abspath(filepath))
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        model_data = {
            "sentence_embeddings": self._sentence_embeddings,
            "similarity_matrix": self._similarity_matrix,
            "sentence_dataframe": self._sentence_df,
            "similarity_threshold": self._similarity_threshold,
            "embedding_model": self.embedding_model,
            "graph": self._G,
        }
        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Loads a trained model from the specified filepath."""
        try:
            with open(filepath, "rb") as f:
                model_data = pickle.load(f)

            self._sentence_embeddings = model_data["sentence_embeddings"]
            self._similarity_matrix = model_data["similarity_matrix"]
            self._sentence_df = model_data["sentence_dataframe"]
            self._similarity_threshold = model_data["similarity_threshold"]
            self.embedding_model = model_data["embedding_model"]
            self._G = model_data["graph"]
            logger.info(f"Model loaded from {filepath}")

        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at {filepath}")
        except Exception as e:
            raise Exception(f"Error loading model from {filepath}: {e}")

    def find_similar_documents(
        self, document_id: Any, top_n: int = 5
    ) -> List[Tuple[Any, float, List[Tuple[int, int, float]]]]:
        """
        Finds the most similar documents to a given document ID based on sentence similarity,
        returning the document ID, similarity score, and matched sentence pairs.

        Args:
            document_id: The ID of the document to find similar documents for.
            top_n: The number of most similar documents to return.

        Returns:
            A list of tuples, where each tuple contains:
            - The document ID of the similar document.
            - The maximum sentence similarity score between the two documents.
            - A list of tuples, where each tuple represents a matched sentence pair
            and contains the sentence positions in the two documents and the similarity score.
        """
        if self._G is None:
            raise ValueError("Graph must be trained before finding similar documents.")

        if document_id not in self._G:
            raise ValueError(f"Document ID '{document_id}' not found in the graph.")

        # Get the neighbors of the document in the graph (i.e., documents with similarity > threshold)
        neighbors = list(self._G.neighbors(document_id))

        # Extract similarity scores and matched sentence pairs for each neighbor
        similarities = []
        for neighbor in neighbors:
            edge_data = self._G[document_id][neighbor]
            max_similarity = edge_data.get("max_sentence_similarity", 0.0)
            matched_sentence_pairs = edge_data.get("matched_sentence_pairs", [])
            similarities.append((neighbor, max_similarity, matched_sentence_pairs))

        # Sort the neighbors by similarity score in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return the top_n most similar documents along with matched sentence pairs
        return similarities[:top_n]

    def query_similar_sentences(
        self, sentence: str, top_n: int = 5
    ) -> List[Tuple[str, float, str]]:
        """
        Finds the most similar sentences to a given query sentence.

        Returns:
            A list of tuples, where each tuple contains:
            - The similar sentence.
            - The similarity score.
            - The document ID of the similar sentence.
        """
        if (
            self._sentence_embeddings is None
            or self._similarity_matrix is None
            or self._sentence_df is None
        ):
            raise ValueError("Model must be trained or loaded before querying.")

        # Embed the query sentence
        query_embedding = self.embedding_model.encode([sentence])[0]

        # Calculate cosine similarities between the query sentence and all sentences in the corpus
        from sklearn.metrics.pairwise import cosine_similarity

        similarities = cosine_similarity([query_embedding], self._sentence_embeddings)[
            0
        ]

        # Get the indices of the top_n most similar sentences
        top_indices = np.argsort(similarities)[-top_n:][::-1]

        results: List[Tuple[str, float, str]] = []
        for idx in top_indices:
            similar_sentence = self._sentence_df.iloc[idx]["sentence"]
            document_id = self._sentence_df.iloc[idx]["document_id"]
            similarity_score = similarities[idx]
            results.append((similar_sentence, similarity_score, document_id))

        return results
