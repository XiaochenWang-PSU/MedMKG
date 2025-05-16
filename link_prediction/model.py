import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# Base Class for Models
class BaseModel(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, init_embeddings=True):
        super(BaseModel, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        if init_embeddings:
            self._init_embeddings()

    def _init_embeddings(self):
        nn.init.uniform_(self.entity_embeddings.weight, -6 / (self.embedding_dim ** 0.5),
                         6 / (self.embedding_dim ** 0.5))
        nn.init.uniform_(self.relation_embeddings.weight, -6 / (self.embedding_dim ** 0.5),
                         6 / (self.embedding_dim ** 0.5))

    def get_embedding(self, heads, relations, tails):
        assert torch.all(heads >= 0) and torch.all(heads < self.num_entities), "Invalid head indices"
        assert torch.all(relations >= 0) and torch.all(relations < self.num_relations), "Invalid relation indices"
        assert torch.all(tails >= 0) and torch.all(tails < self.num_entities), "Invalid tail indices"
    
        head_emb = self.entity_embeddings(heads)
        relation_emb = self.relation_embeddings(relations)
        tail_emb = self.entity_embeddings(tails)
        return head_emb, relation_emb, tail_emb



# TransR Model
class TransR(BaseModel):
    def __init__(self, num_entities, num_relations, embedding_dim, relation_dim, margin=1.0, p_norm=2):
        """
        TransR Model: Projects entities into a relation-specific space.
        Args:
            relation_dim: Dimension of the relation space (can differ from entity space).
        """
        super(TransR, self).__init__(num_entities, num_relations, embedding_dim, init_embeddings=False)
        self.relation_dim = relation_dim
        self.entity_proj_matrix = nn.Embedding(num_relations, embedding_dim * relation_dim)
        self.relation_embeddings = nn.Embedding(num_relations, relation_dim)
        self.margin = margin
        self.p_norm = p_norm
        self.loss = nn.MarginRankingLoss(margin=margin)

        self._init_embeddings()

    def _init_embeddings(self):
        # Initialize embeddings
        super()._init_embeddings()
        nn.init.xavier_uniform_(self.entity_proj_matrix.weight)

    def forward(self, pos_triplets, neg_triplets):
        pos_scores = self._score_triplets(pos_triplets)
        neg_scores = self._score_triplets(neg_triplets)
        target = torch.ones_like(pos_scores)
        return self.loss(pos_scores, neg_scores, target)

    def _project_to_relation_space(self, entity_emb, proj_matrix):
        """Project entities into relation-specific space."""
        proj_matrix = proj_matrix.view(-1, self.embedding_dim, self.relation_dim)  # Reshape projection matrix
        return torch.bmm(entity_emb.unsqueeze(1), proj_matrix).squeeze(1)

    def _score_triplets(self, triplets):
        """
        Compute scores for triplets after projecting entities into relation-specific space.
        """
        heads, relations, tails = triplets[:, 0], triplets[:, 1], triplets[:, 2]
        head_emb = self.entity_embeddings(heads)
        tail_emb = self.entity_embeddings(tails)
        relation_emb = self.relation_embeddings(relations)
        proj_matrix = self.entity_proj_matrix(relations)

        # Project head and tail embeddings into relation space
        head_proj = self._project_to_relation_space(head_emb, proj_matrix)
        tail_proj = self._project_to_relation_space(tail_emb, proj_matrix)

        # Compute score in relation space
        return -torch.norm(head_proj + relation_emb - tail_proj, p=self.p_norm, dim=1)

# TransE
class TransE(BaseModel):
    def __init__(self, num_entities, num_relations, embedding_dim, margin=1.0):
        super(TransE, self).__init__(num_entities, num_relations, embedding_dim)
        self.margin = margin
        self.loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, pos_triplets, neg_triplets):
        pos_scores = self._score_triplets(pos_triplets)
        neg_scores = self._score_triplets(neg_triplets)
        target = torch.ones_like(pos_scores)
        return self.loss(pos_scores, neg_scores, target)

    def _score_triplets(self, triplets):
        heads, relations, tails = triplets[:, 0], triplets[:, 1], triplets[:, 2]
        head_emb, relation_emb, tail_emb = self.get_embedding(heads, relations, tails)
        return -torch.norm(head_emb + relation_emb - tail_emb, p=2, dim=1)


# DistMult
class DistMult(BaseModel):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(DistMult, self).__init__(num_entities, num_relations, embedding_dim)

    def forward(self, pos_triplets, neg_triplets):
        pos_scores = self._score_triplets(pos_triplets)
        neg_scores = self._score_triplets(neg_triplets)
        target = torch.ones_like(pos_scores)
        return nn.MarginRankingLoss(margin=1.0)(pos_scores, neg_scores, target)

    def _score_triplets(self, triplets):
        heads, relations, tails = triplets[:, 0], triplets[:, 1], triplets[:, 2]
        head_emb, relation_emb, tail_emb = self.get_embedding(heads, relations, tails)
        return torch.sum(head_emb * relation_emb * tail_emb, dim=1)


# ComplEx
class ComplEx(BaseModel):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(ComplEx, self).__init__(num_entities, num_relations, embedding_dim)
        self.embedding_dim = embedding_dim // 2

    def forward(self, pos_triplets, neg_triplets):
        pos_scores = self._score_triplets(pos_triplets)
        neg_scores = self._score_triplets(neg_triplets)
        target = torch.ones_like(pos_scores)
        return torch.mean(F.relu(1.0 - pos_scores + neg_scores))

    def _get_complex_embedding(self, embeddings):
        return embeddings[:, :self.embedding_dim], embeddings[:, self.embedding_dim:]

    def _score_triplets(self, triplets):
        heads, relations, tails = triplets[:, 0], triplets[:, 1], triplets[:, 2]
        head_emb, relation_emb, tail_emb = self.get_embedding(heads, relations, tails)
        head_re, head_im = self._get_complex_embedding(head_emb)
        relation_re, relation_im = self._get_complex_embedding(relation_emb)
        tail_re, tail_im = self._get_complex_embedding(tail_emb)

        return torch.sum(
            head_re * relation_re * tail_re
            + head_re * relation_im * tail_im
            + head_im * relation_re * tail_im
            - head_im * relation_im * tail_re,
            dim=1
        )


# RESCAL
# RESCAL
class RESCAL(BaseModel):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(RESCAL, self).__init__(num_entities, num_relations, embedding_dim)
        self.relation_matrices = nn.Parameter(torch.randn(num_relations, embedding_dim, embedding_dim))

    def forward(self, pos_triplets, neg_triplets):
        pos_scores = self._score_triplets(pos_triplets)
        neg_scores = self._score_triplets(neg_triplets)
        target = torch.ones_like(pos_scores)
        return nn.MarginRankingLoss(margin=1.0)(pos_scores, neg_scores, target)

    def _score_triplets(self, triplets):
        heads, relations, tails = triplets[:, 0], triplets[:, 1], triplets[:, 2]
        head_emb, _, tail_emb = self.get_embedding(heads, relations, tails)
        relation_mat = self.relation_matrices[relations]

        # Handle batch processing safely
        scores = torch.bmm(head_emb.unsqueeze(1), relation_mat)
        scores = torch.bmm(scores, tail_emb.unsqueeze(2)).squeeze()
        return scores


class TransD(BaseModel):
    def __init__(self, num_entities, num_relations, embedding_dim, margin=1.0, p_norm=2):
        """
        TransD Model: Dynamic projection-based method with modular BaseModel.
        Args:
            num_entities (int): Number of unique entities.
            num_relations (int): Number of unique relations.
            embedding_dim (int): Dimension of entity and relation embeddings.
            margin (float): Margin for the ranking loss.
            p_norm (int): Norm degree (L1 or L2).
        """
        super(TransD, self).__init__(num_entities, num_relations, embedding_dim)
        self.margin = margin
        self.p_norm = p_norm
        self.loss_fn = nn.MarginRankingLoss(margin=margin)

        # Projection vectors for entities and relations
        self.entity_proj = nn.Embedding(num_entities, embedding_dim)
        self.relation_proj = nn.Embedding(num_relations, embedding_dim)

        # Initialize projection vectors
        self._init_proj_embeddings()

    def _init_proj_embeddings(self):
        """Initialize projection vectors uniformly."""
        nn.init.xavier_uniform_(self.entity_proj.weight)
        nn.init.xavier_uniform_(self.relation_proj.weight)

    def _project_to_relation_space(self, entity_emb, entity_proj, relation_proj):
        """
        Project entity embeddings into relation-specific space.
        Args:
            entity_emb: Base entity embeddings.
            entity_proj: Entity projection vectors.
            relation_proj: Relation projection vectors.
        Returns:
            Projected entity embeddings.
        """
        scaling_factor = torch.sum(entity_proj * relation_proj, dim=1, keepdim=True)  # Scalar per sample
        return entity_emb + scaling_factor * entity_proj  # Lightweight projection

    def _score_triplets(self, triplets):
        """
        Compute the scores for a batch of triplets.
        Args:
            triplets: Tensor of triplets [heads, relations, tails].
        Returns:
            Tensor: Triplet scores.
        """
        heads, relations, tails = triplets[:, 0], triplets[:, 1], triplets[:, 2]

        # Fetch embeddings
        head_emb, relation_emb, tail_emb = self.get_embedding(heads, relations, tails)

        # Fetch projection vectors
        head_proj = self.entity_proj(heads)
        tail_proj = self.entity_proj(tails)
        relation_proj = self.relation_proj(relations)

        # Project head and tail embeddings into relation-specific space
        head_proj_emb = self._project_to_relation_space(head_emb, head_proj, relation_proj)
        tail_proj_emb = self._project_to_relation_space(tail_emb, tail_proj, relation_proj)

        # Compute distance-based score in relation space
        return -torch.norm(head_proj_emb + relation_emb - tail_proj_emb, p=self.p_norm, dim=1)

    def forward(self, pos_triplets, neg_triplets):
        """
        Compute the loss for a batch of positive and negative triplets.
        Args:
            pos_triplets: Positive triplets (batch_size, 3).
            neg_triplets: Negative triplets (batch_size, 3).
        Returns:
            Margin ranking loss.
        """
        pos_scores = self._score_triplets(pos_triplets)
        neg_scores = self._score_triplets(neg_triplets)

        # Target: positive scores should be higher than negative scores
        target = torch.ones_like(pos_scores)
        return self.loss_fn(pos_scores, neg_scores, target)


### NTN
#class NTN(BaseModel):
#    def __init__(self, num_entities, num_relations, embedding_dim, tensor_dim=10):
#        super(NTN, self).__init__(num_entities, num_relations, embedding_dim)
#        self.relation_tensors = nn.Parameter(torch.randn(num_relations, tensor_dim, embedding_dim, embedding_dim))
#        self.relation_bias = nn.Parameter(torch.randn(num_relations, tensor_dim))
#        self.relation_weights = nn.Parameter(torch.randn(num_relations, tensor_dim, embedding_dim))
#
#    def forward(self, pos_triplets, neg_triplets):
#        pos_scores = self._score_triplets(pos_triplets)
#        neg_scores = self._score_triplets(neg_triplets)
#        target = torch.ones_like(pos_scores)
#        return nn.MarginRankingLoss(margin=1.0)(pos_scores, neg_scores, target)
#
#    def _score_triplets(self, triplets):
#        heads, relations, tails = triplets[:, 0], triplets[:, 1], triplets[:, 2]
#        head_emb, _, tail_emb = self.get_embedding(heads, relations, tails)
#
#        # Bilinear transformation
#        r_tensor = self.relation_tensors[relations]
#        bilinear_score = torch.einsum('bi,btij,bj->bt', head_emb, r_tensor, tail_emb)
#
#        # Linear transformation
#        linear_score = torch.einsum('btj,bj->bt', self.relation_weights[relations], head_emb * tail_emb)
#
#        # Combine scores
#        return torch.sum(bilinear_score + linear_score + self.relation_bias[relations], dim=1)


class NTN(BaseModel):
    def __init__(self, num_entities, num_relations, embedding_dim, tensor_dim=10):
        super(NTN, self).__init__(num_entities, num_relations, embedding_dim)
        self.tensor_dim = tensor_dim
        # Relation-specific tensor slices: shape (num_relations, tensor_dim, embedding_dim, embedding_dim)
        self.relation_tensors = nn.Parameter(torch.randn(num_relations, tensor_dim, embedding_dim, embedding_dim))
        # Relation-specific bias: shape (num_relations, tensor_dim)
        self.relation_bias = nn.Parameter(torch.randn(num_relations, tensor_dim))
        # Linear term for concatenated entity embeddings: shape (num_relations, tensor_dim, 2 * embedding_dim)
        self.relation_V = nn.Parameter(torch.randn(num_relations, tensor_dim, 2 * embedding_dim))
        # Final scoring vector: shape (num_relations, tensor_dim)
        self.relation_u = nn.Parameter(torch.randn(num_relations, tensor_dim))
    
    def forward(self, pos_triplets, neg_triplets):
        pos_scores = self._score_triplets(pos_triplets)
        neg_scores = self._score_triplets(neg_triplets)
        target = torch.ones_like(pos_scores)
        return nn.MarginRankingLoss(margin=1.0)(pos_scores, neg_scores, target)
    
    def _score_triplets(self, triplets):
        heads, relations, tails = triplets[:, 0], triplets[:, 1], triplets[:, 2]
        head_emb, _, tail_emb = self.get_embedding(heads, relations, tails)
        # Concatenate head and tail embeddings: shape (batch, 2*embedding_dim)
        ht_cat = torch.cat([head_emb, tail_emb], dim=1)
        # Get relation-specific parameters for each sample
        rel_tensors = self.relation_tensors[relations]  # (batch, tensor_dim, d, d)
        rel_bias = self.relation_bias[relations]          # (batch, tensor_dim)
        rel_V = self.relation_V[relations]                # (batch, tensor_dim, 2*d)
        rel_u = self.relation_u[relations]                # (batch, tensor_dim)
        # Bilinear term: for each slice i, compute h^T * W_r[i] * t
        bilinear = torch.einsum('bi,btij,bj->bt', head_emb, rel_tensors, tail_emb)  # (batch, tensor_dim)
        # Linear term: V_r * [h;t]
        linear_term = torch.bmm(rel_V, ht_cat.unsqueeze(2)).squeeze(2)  # (batch, tensor_dim)
        # Pre-activation and non-linearity
        preact = bilinear + linear_term + rel_bias  # (batch, tensor_dim)
        activated = torch.tanh(preact)
        # Final score: dot with relation-specific vector u
        score = torch.sum(rel_u * activated, dim=1)  # (batch,)
        return score

# SimplE
class SimplE(BaseModel):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(SimplE, self).__init__(num_entities, num_relations, embedding_dim, init_embeddings=False)
        
        # Define SimplE-specific components
        self.relation_inv_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.entity_head_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.entity_tail_embeddings = nn.Embedding(num_entities, embedding_dim)
        
        # Initialize weights explicitly
        self._init_embeddings()

    def _init_embeddings(self):
        # Initialize embeddings with uniform distribution
        nn.init.uniform_(self.entity_head_embeddings.weight, -6 / (self.embedding_dim ** 0.5), 6 / (self.embedding_dim ** 0.5))
        nn.init.uniform_(self.entity_tail_embeddings.weight, -6 / (self.embedding_dim ** 0.5), 6 / (self.embedding_dim ** 0.5))
        nn.init.uniform_(self.relation_embeddings.weight, -6 / (self.embedding_dim ** 0.5), 6 / (self.embedding_dim ** 0.5))
        nn.init.uniform_(self.relation_inv_embeddings.weight, -6 / (self.embedding_dim ** 0.5), 6 / (self.embedding_dim ** 0.5))

    def forward(self, pos_triplets, neg_triplets):
        pos_scores = self._score_triplets(pos_triplets)
        neg_scores = self._score_triplets(neg_triplets)
        target = torch.ones_like(pos_scores)
        return torch.mean(F.relu(1.0 - pos_scores + neg_scores))

    def _score_triplets(self, triplets):
        heads, relations, tails = triplets[:, 0], triplets[:, 1], triplets[:, 2]

        # Fetch embeddings
        head_emb = self.entity_head_embeddings(heads)  # Head entity embedding
        tail_emb = self.entity_tail_embeddings(tails)  # Tail entity embedding
        rel_emb = self.relation_embeddings(relations)  # Forward relation embedding
        rel_inv_emb = self.relation_inv_embeddings(relations)  # Inverse relation embedding

        # Compute SimplE score
        forward_score = torch.sum(head_emb * rel_emb * tail_emb, dim=1)
        inverse_score = torch.sum(tail_emb * rel_inv_emb * head_emb, dim=1)

        return (forward_score + inverse_score) / 2



class TuckER(BaseModel):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(TuckER, self).__init__(num_entities, num_relations, embedding_dim)
        self.core_tensor = nn.Parameter(torch.randn(embedding_dim, embedding_dim, embedding_dim))
        self.input_dropout = nn.Dropout(0.3)
        self.hidden_dropout = nn.Dropout(0.3)
        self.output_dropout = nn.Dropout(0.3)
        self.init()

    def init(self):
        """Initialize the core tensor and embeddings."""
        nn.init.xavier_uniform_(self.core_tensor)
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def forward(self, pos_triplets, neg_triplets):
        pos_scores = self._score_triplets(pos_triplets)
        neg_scores = self._score_triplets(neg_triplets)
        target = torch.ones_like(pos_scores)
        return torch.mean(F.relu(1.0 - pos_scores + neg_scores))

    def _score_triplets(self, triplets):
        heads, relations, tails = triplets[:, 0], triplets[:, 1], triplets[:, 2]

        # Apply dropout to entity and relation embeddings
        head_emb = self.input_dropout(self.entity_embeddings(heads))
        relation_emb = self.input_dropout(self.relation_embeddings(relations))
        tail_emb = self.output_dropout(self.entity_embeddings(tails))

        # Tucker decomposition with core tensor
        relation_core = torch.einsum("ijk,bi->bjk", self.core_tensor, relation_emb)  # Core tensor interaction
        output = torch.einsum("bjk,bj->bk", relation_core, head_emb)  # Interaction with head embedding

        # Return score as dot product with tail embedding
        return torch.sum(output * tail_emb, dim=1)
        
class ConvE(BaseModel):
    def __init__(self, num_entities, num_relations, embedding_dim, init_embeddings=True):
        super(ConvE, self).__init__(num_entities, num_relations, embedding_dim, init_embeddings)
        assert embedding_dim % 2 == 0, "Embedding dimension must be even for ConvE reshaping."

        # Define convolutional parameters
        self.embedding_dim = embedding_dim  # Do not divide by 2
        self.input_dropout = nn.Dropout(0.2)
        self.feature_map_dropout = nn.Dropout(0.2)
        self.hidden_dropout = nn.Dropout(0.3)

        # Convolutional layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        # Dynamically calculate the number of features after convolution
        self.conv_output_size = self._calculate_conv_output_size()
        self.fc = nn.Linear(self.conv_output_size, embedding_dim)

    def _calculate_conv_output_size(self):
        """Calculate the number of features after the convolutional layer."""
        dummy_input = torch.zeros(1, 1, self.embedding_dim, 2)  # Simulate input to convolution
        
        dummy_output = self.conv1(dummy_input)  # Pass through convolution
        dummy_output = dummy_output.view(dummy_output.size(0), -1)  # Flatten the output
        return dummy_output.size(1)  # Return the flattened size


    def _score_triplets(self, triplets):
        # Extract embeddings
        heads, relations, tails = triplets[:, 0], triplets[:, 1], triplets[:, 2]
        head_emb, relation_emb, tail_emb = self.get_embedding(heads, relations, tails)
    
    
        # Correct reshaping
        assert head_emb.size(1) == self.embedding_dim, "Mismatch between embedding size and self.embedding_dim"
        head_reshaped = head_emb.view(head_emb.size(0), 1, self.embedding_dim // 4, 4)
        relation_reshaped = relation_emb.view(relation_emb.size(0), 1, self.embedding_dim // 4, 4)
    
       
        # Concatenate along the feature dimension
        x = torch.cat([head_reshaped, relation_reshaped], dim=3)
        
        # Convolution and further processing
        x = self.input_dropout(x)
        x = self.conv1(x)
        x = self.feature_map_dropout(F.relu(x))
        x = x.view(x.size(0), -1)  # Flatten for fully connected layer
        x = self.hidden_dropout(self.fc(x))
    
        # Ensure consistency
        assert x.size(0) == tail_emb.size(0), "Batch size mismatch between x and tail_emb!"
    
        # Compute scores
        score = torch.sum(x * tail_emb, dim=1)
        return score
    


    





    def forward(self, pos_triplets, neg_triplets):
        # Concatenate positive and negative triplets
        all_triplets = torch.cat([pos_triplets, neg_triplets], dim=0)

    
        # Compute scores for all triplets
        all_scores = self._score_triplets(all_triplets)
    
        # Separate scores for positive and negative triplets
        num_pos = pos_triplets.size(0)
        pos_scores, neg_scores = all_scores[:num_pos], all_scores[num_pos:]

    
        # Compute loss
        target = torch.ones_like(pos_scores)
        return nn.MarginRankingLoss(margin=1.0)(pos_scores, neg_scores, target)






# ConvR
class ConvR(BaseModel):
    def __init__(self, num_entities, num_relations, embedding_dim, init_embeddings=True):
        super(ConvR, self).__init__(num_entities, num_relations, embedding_dim, init_embeddings)
        assert embedding_dim > 3, "Embedding dimension must be larger than the kernel size to avoid size mismatch."
        
        # Convolutional layer
        self.conv = nn.Conv2d(1, 32, (3, 3), stride=1, padding=1)  # Padding added
        self.dropout = nn.Dropout(0.3)

        # Dynamically calculate the input size for the fully connected layer
        self.conv_output_size = 32 * embedding_dim
        self.fc = nn.Linear(self.conv_output_size, embedding_dim)

    def _score_triplets(self, triplets):
        heads, relations, tails = triplets[:, 0], triplets[:, 1], triplets[:, 2]
        head_emb, relation_emb, tail_emb = self.get_embedding(heads, relations, tails)

        # Reshape and apply convolution
        x = head_emb * relation_emb  # Element-wise product
        x = x.view(-1, 1, self.embedding_dim, 1)  # Reshape for convolution
        x = self.conv(x)  # Apply convolution
        x = F.relu(x)
        x = x.view(x.shape[0], -1)  # Flatten output
        x = self.fc(self.dropout(x))  # Fully connected layer
        score = torch.sum(x * tail_emb, dim=1)  # Dot product for scoring
        return score

    def forward(self, pos_triplets, neg_triplets):
        pos_scores = self._score_triplets(pos_triplets)
        neg_scores = self._score_triplets(neg_triplets)
        target = torch.ones_like(pos_scores)
        return nn.MarginRankingLoss(margin=1.0)(pos_scores, neg_scores, target)




### HypER
#class HypER(BaseModel):
#    def __init__(self, num_entities, num_relations, embedding_dim, init_embeddings=True):
#        super(HypER, self).__init__(num_entities, num_relations, embedding_dim, init_embeddings)
#        self.fc = nn.Linear(embedding_dim, embedding_dim)
#        self.dropout = nn.Dropout(0.3)
#
#    def _score_triplets(self, triplets):
#        heads, relations, tails = triplets[:, 0], triplets[:, 1], triplets[:, 2]
#        head_emb, relation_emb, tail_emb = self.get_embedding(heads, relations, tails)
#        relation_trans = self.fc(relation_emb)
#        x = head_emb * relation_trans
#        score = torch.sum(x * tail_emb, dim=1)  # Dot product for scoring
#        return score
#
#    def forward(self, pos_triplets, neg_triplets):
#        pos_scores = self._score_triplets(pos_triplets)
#        neg_scores = self._score_triplets(neg_triplets)
#        target = torch.ones_like(pos_scores)
#        return nn.MarginRankingLoss(margin=1.0)(pos_scores, neg_scores, target)

class HypER(BaseModel):
    def __init__(self, num_entities, num_relations, embedding_dim, filter_size=3, num_filters=32):
        super(HypER, self).__init__(num_entities, num_relations, embedding_dim, init_embeddings=True)
        self.filter_size = filter_size
        self.num_filters = num_filters
        # Hypernetwork: generate filter weights from relation embeddings.
        # Each filter weight tensor is of shape (num_filters, 1, filter_size, filter_size)
        self.hyper_linear = nn.Linear(embedding_dim, num_filters * filter_size * filter_size)
        # Assume the head embedding will be reshaped into a square: embedding_dim = h * w
        self.h = int(embedding_dim**0.5)
        self.w = self.h
        assert self.h * self.w == embedding_dim, "embedding_dim must be a perfect square for HypER."
        # Fully connected layer to project convolved features back to embedding_dim
        self.fc = nn.Linear(num_filters * self.h * self.w, embedding_dim)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, pos_triplets, neg_triplets):
        pos_scores = self._score_triplets(pos_triplets)
        neg_scores = self._score_triplets(neg_triplets)
        target = torch.ones_like(pos_scores)
        return nn.MarginRankingLoss(margin=1.0)(pos_scores, neg_scores, target)
    
    def _score_triplets(self, triplets):
        heads, relations, tails = triplets[:, 0], triplets[:, 1], triplets[:, 2]
        head_emb, relation_emb, tail_emb = self.get_embedding(heads, relations, tails)
        # Reshape head embeddings into 2D grids: (batch, 1, h, w)
        head_reshaped = head_emb.view(-1, 1, self.h, self.w)
        batch_size = relation_emb.size(0)
        # Generate convolution filter weights for each sample
        filter_weights = self.hyper_linear(relation_emb)  # (batch, num_filters * filter_size * filter_size)
        filter_weights = filter_weights.view(batch_size, self.num_filters, 1, self.filter_size, self.filter_size)
        # Apply convolution with sample-specific filters (using a loop for clarity)
        conv_outputs = []
        for i in range(batch_size):
            # Use padding to preserve spatial dimensions
            conv_out = F.conv2d(head_reshaped[i:i+1], weight=filter_weights[i], padding=self.filter_size // 2)
            conv_outputs.append(conv_out)
        conv_outputs = torch.cat(conv_outputs, dim=0)  # (batch, num_filters, h, w)
        # Flatten and apply dropout and fully connected layer
        conv_flat = conv_outputs.view(batch_size, -1)
        conv_flat = self.dropout(conv_flat)
        x = self.fc(conv_flat)  # (batch, embedding_dim)
        # Final score via dot product with tail embedding
        score = torch.sum(x * tail_emb, dim=1)
        return score
# R-GCN
#class RGCN(BaseModel):
#    def __init__(self, num_entities, num_relations, embedding_dim, init_embeddings=True):
#        super(RGCN, self).__init__(num_entities, num_relations, embedding_dim, init_embeddings)
#        self.gcn = nn.Linear(embedding_dim, embedding_dim)
#        self.dropout = nn.Dropout(0.3)
#
#    def _score_triplets(self, triplets):
#        heads, relations, tails = triplets[:, 0], triplets[:, 1], triplets[:, 2]
#        head_emb, relation_emb, tail_emb = self.get_embedding(heads, relations, tails)
#        head_emb = self.gcn(head_emb)
#        relation_emb = self.gcn(relation_emb)
#        tail_emb = self.gcn(tail_emb)
#        score = -torch.norm(head_emb + relation_emb - tail_emb, p=2, dim=1)
#        return score
#
#    def forward(self, pos_triplets, neg_triplets):
#        pos_scores = self._score_triplets(pos_triplets)
#        neg_scores = self._score_triplets(neg_triplets)
#        target = torch.ones_like(pos_scores)
#        return nn.MarginRankingLoss(margin=1.0)(pos_scores, neg_scores, target)


class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_relations, activation=F.relu, dropout=0.3):
        super(RGCNLayer, self).__init__()
        self.num_relations = num_relations
        self.in_feat = in_feat
        self.out_feat = out_feat
        # Weight for each relation type
        self.weight = nn.Parameter(torch.Tensor(num_relations, in_feat, out_feat))
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, entity_embeddings, edge_index, edge_type):
        """
        entity_embeddings: (num_entities, in_feat)
        edge_index: (2, num_edges) with [source_indices; target_indices]
        edge_type: (num_edges,) with relation type for each edge
        """
        num_entities = entity_embeddings.size(0)
        messages = torch.zeros(num_entities, self.out_feat, device=entity_embeddings.device)
        # For each relation type, aggregate messages from neighbors.
        for r in range(self.num_relations):
            mask = (edge_type == r)
            if mask.sum() == 0:
                continue
            src = edge_index[0, mask]
            tgt = edge_index[1, mask]
            msg = torch.matmul(entity_embeddings[src], self.weight[r])
            messages.index_add_(0, tgt, msg)
        out = self.activation(messages)
        out = self.dropout(out)
        return out
class RGCN(BaseModel):
    def __init__(self, num_entities, num_relations, embedding_dim, edge_index=None, edge_type=None):
        super(RGCN, self).__init__(num_entities, num_relations, embedding_dim, init_embeddings=True)
        self.rgcn_layer = RGCNLayer(embedding_dim, embedding_dim, num_relations, activation=F.relu, dropout=0.3)
        self.edge_index = edge_index
        self.edge_type = edge_type
 
    def forward(self, pos_triplets, neg_triplets, edge_index=None, edge_type=None):
        # Use provided edge_index and edge_type if given, otherwise use the stored ones.
        if edge_index is None or edge_type is None:
            if self.edge_index is None or self.edge_type is None:
                raise ValueError("Graph structure (edge_index and edge_type) must be provided.")
            edge_index, edge_type = self.edge_index, self.edge_type
        updated_entity_embeddings = self.rgcn_layer(self.entity_embeddings.weight, edge_index, edge_type)
        pos_scores = self._score_triplets(pos_triplets, updated_entity_embeddings)
        neg_scores = self._score_triplets(neg_triplets, updated_entity_embeddings)
        target = torch.ones_like(pos_scores)
        return nn.MarginRankingLoss(margin=1.0)(pos_scores, neg_scores, target)
 
    def _score_triplets(self, triplets, entity_embeddings):
        heads, relations, tails = triplets[:, 0], triplets[:, 1], triplets[:, 2]
        head_emb = entity_embeddings[heads]
        relation_emb = self.relation_embeddings(relations)
        tail_emb = entity_embeddings[tails]
        score = -torch.norm(head_emb + relation_emb - tail_emb, p=2, dim=1)
        return score

# AttH
class AttH(BaseModel):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(AttH, self).__init__(num_entities, num_relations, embedding_dim)
        self.hyperplane = nn.Parameter(torch.Tensor(embedding_dim))
        nn.init.uniform_(self.hyperplane, -6 / (embedding_dim ** 0.5), 6 / (embedding_dim ** 0.5))
        self.attention = nn.Linear(embedding_dim * 3, embedding_dim)

    def forward(self, pos_triplets, neg_triplets):
        pos_scores = self._score_triplets(pos_triplets)
        neg_scores = self._score_triplets(neg_triplets)
        target = torch.ones_like(pos_scores)
        return nn.MarginRankingLoss(margin=1.0)(pos_scores, neg_scores, target)

    def _score_triplets(self, triplets):
        heads, relations, tails = triplets[:, 0], triplets[:, 1], triplets[:, 2]
        head_emb, relation_emb, tail_emb = self.get_embedding(heads, relations, tails)

        # Compute attention
        combined = torch.cat([head_emb, relation_emb, self.hyperplane.expand_as(head_emb)], dim=-1)
        attention_weights = torch.softmax(self.attention(combined), dim=-1)
        weighted_head = head_emb * attention_weights

        # Scoring
        score = -torch.norm(weighted_head + relation_emb - tail_emb, p=2, dim=1)
        return score


class RotatE(BaseModel):
    def __init__(self, num_entities, num_relations, embedding_dim, init_embeddings=True):
        super(RotatE, self).__init__(num_entities, num_relations, embedding_dim, init_embeddings)
        assert embedding_dim % 2 == 0, "Embedding dimension must be even for RotatE."
        self.embedding_dim = embedding_dim // 2  # Allocate half for real and imaginary parts

    def _score_triplets(self, triplets):
        heads, relations, tails = triplets[:, 0], triplets[:, 1], triplets[:, 2]
        head_emb, relation_emb, tail_emb = self.get_embedding(heads, relations, tails)

        # Split embeddings into real and imaginary parts
        head_re, head_im = head_emb[..., :self.embedding_dim], head_emb[..., self.embedding_dim:]
        tail_re, tail_im = tail_emb[..., :self.embedding_dim], tail_emb[..., self.embedding_dim:]
        rel_phase = relation_emb[..., :self.embedding_dim]  # Use only the first half for phase

        # Compute rotation
        rotated_re = head_re * torch.cos(rel_phase) - head_im * torch.sin(rel_phase)
        rotated_im = head_re * torch.sin(rel_phase) + head_im * torch.cos(rel_phase)

        # Compute scores using Euclidean distance
        score_re = rotated_re - tail_re
        score_im = rotated_im - tail_im
        score = -torch.norm(torch.cat([score_re, score_im], dim=-1), p=2, dim=-1)
        return score

    def forward(self, pos_triplets, neg_triplets):
        pos_scores = self._score_triplets(pos_triplets)
        neg_scores = self._score_triplets(neg_triplets)
        target = torch.ones_like(pos_scores)
        return nn.MarginRankingLoss(margin=1.0)(pos_scores, neg_scores, target)

## ConvRot
#class ConvRot(BaseModel):
#    def __init__(self, num_entities, num_relations, embedding_dim, init_embeddings=True):
#        super(ConvRot, self).__init__(num_entities, num_relations, embedding_dim, init_embeddings)
#        self.phase_shift = nn.Parameter(torch.Tensor(embedding_dim))
#        nn.init.uniform_(self.phase_shift, -6 / (embedding_dim ** 0.5), 6 / (embedding_dim ** 0.5))
#
#    def forward(self, heads, relations, tails):
#        head_emb, relation_emb, tail_emb = self.get_embedding(heads, relations, tails)
#        phase = relation_emb * self.phase_shift
#        rotated_head = head_emb * torch.cos(phase) + torch.sin(phase)
#        score = torch.sum(rotated_head * tail_emb, dim=-1)
#        return score

# TransH
class TransH(BaseModel):
    """
    TransH: Knowledge Graph Embedding with Hyperplane Projection
    Reference: Wang, Z., Zhang, J., Feng, J., & Chen, Z. (2014). Knowledge Graph Embedding by Translating on Hyperplanes.
    """
    def __init__(self, num_entities, num_relations, embedding_dim, margin=1.0, p_norm=2):
        """
        Args:
            num_entities: Total number of entities.
            num_relations: Total number of relations.
            embedding_dim: Dimension of embeddings.
            margin: Margin for margin-based ranking loss.
            p_norm: Norm type for scoring (L1 or L2).
        """
        super(TransH, self).__init__(num_entities, num_relations, embedding_dim, init_embeddings=False)
        self.margin = margin
        self.p_norm = p_norm
        self.norm_embeddings = nn.Embedding(num_relations, embedding_dim)  # Normal vector for hyperplanes
        self.loss = nn.MarginRankingLoss(margin=margin)

        self._init_embeddings()

    def _init_embeddings(self):
        # Initialize embeddings
        super()._init_embeddings()
        nn.init.uniform_(self.norm_embeddings.weight, -6 / (self.embedding_dim ** 0.5), 6 / (self.embedding_dim ** 0.5))

    def forward(self, pos_triplets, neg_triplets):
        """
        Compute the margin ranking loss for positive and negative triplets.
        """
        pos_scores = self._score_triplets(pos_triplets)
        neg_scores = self._score_triplets(neg_triplets)
        target = torch.ones_like(pos_scores)
        return self.loss(pos_scores, neg_scores, target)

    def _project_to_hyperplane(self, entity_emb, norm_emb):
        """
        Project entity embeddings onto the relation-specific hyperplane.
        """
        return entity_emb - torch.sum(entity_emb * norm_emb, dim=-1, keepdim=True) * norm_emb

    def _score_triplets(self, triplets):
        """
        Compute scores for triplets.
        """
        heads, relations, tails = triplets[:, 0], triplets[:, 1], triplets[:, 2]
        head_emb, relation_emb, tail_emb = self.get_embedding(heads, relations, tails)
        norm_emb = F.normalize(self.norm_embeddings(relations), p=2, dim=-1)  # Normalize the hyperplane normals

        # Project entities onto the hyperplane
        head_proj = self._project_to_hyperplane(head_emb, norm_emb)
        tail_proj = self._project_to_hyperplane(tail_emb, norm_emb)

        # Compute the score
        return -torch.norm(head_proj + relation_emb - tail_proj, p=self.p_norm, dim=-1)

## MurP
#class MurP(BaseModel):
#    def __init__(self, num_entities, num_relations, embedding_dim, c=1.0):
#        super(MurP, self).__init__(num_entities, num_relations, embedding_dim)
#        self.c = c  # Curvature of the hyperbolic space
#
#    def mobius_addition(self, x, y):
#        xy = torch.sum(x * y, dim=-1, keepdim=True)
#        x2 = torch.sum(x**2, dim=-1, keepdim=True)
#        y2 = torch.sum(y**2, dim=-1, keepdim=True)
#        denominator = 1 + 2 * self.c * xy + self.c**2 * x2 * y2
#        return (x + y + self.c * xy * (x + y)) / denominator
#
#    def forward(self, pos_triplets, neg_triplets):
#        pos_scores = self._score_triplets(pos_triplets)
#        neg_scores = self._score_triplets(neg_triplets)
#        target = torch.ones_like(pos_scores)
#        return nn.MarginRankingLoss(margin=1.0)(pos_scores, neg_scores, target)
#
#    def _score_triplets(self, triplets):
#        heads, relations, tails = triplets[:, 0], triplets[:, 1], triplets[:, 2]
#        head_emb, relation_emb, tail_emb = self.get_embedding(heads, relations, tails)
#
#         Hyperbolic scoring using Möbius addition
#        head_trans = self.mobius_addition(head_emb, relation_emb)
#        score = -torch.norm(head_trans - tail_emb, p=2, dim=1)
#        return score
class MurP(BaseModel):
    def __init__(self, num_entities, num_relations, embedding_dim, c=1.0):
        super(MurP, self).__init__(num_entities, num_relations, embedding_dim, init_embeddings=True)
        self.c = c  # curvature parameter
    
    def mobius_addition(self, x, y):
    
        """
        Implements Möbius addition in the Poincaré ball:
        x ?_c y = ((1 + 2c?x,y? + c?y?²) x + (1 - c?x?²) y) / (1 + 2c?x,y? + c²?x?²?y?²)
        """
        
        c = self.c
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        x2 = torch.sum(x ** 2, dim=-1, keepdim=True)
        y2 = torch.sum(y ** 2, dim=-1, keepdim=True)
        numerator = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denominator = 1 + 2 * c * xy + (c ** 2) * x2 * y2
        return numerator / denominator
    
    def forward(self, pos_triplets, neg_triplets):
        pos_scores = self._score_triplets(pos_triplets)
        neg_scores = self._score_triplets(neg_triplets)
        target = torch.ones_like(pos_scores)
        return nn.MarginRankingLoss(margin=1.0)(pos_scores, neg_scores, target)
    
    def _score_triplets(self, triplets):
        heads, relations, tails = triplets[:, 0], triplets[:, 1], triplets[:, 2]
        head_emb, relation_emb, tail_emb = self.get_embedding(heads, relations, tails)
        # Translate head in hyperbolic space using Möbius addition
        head_trans = self.mobius_addition(head_emb, relation_emb)
        score = -torch.norm(head_trans - tail_emb, p=2, dim=1)
        return score

# MurE
class MurE(BaseModel):
    """
    MurE: Knowledge Graph Embedding in Euclidean Space with Translation Scoring
    Reference: Balazevic et al., 2019
    """

    def __init__(self, num_entities, num_relations, embedding_dim, p_norm=2, init_embeddings=True):
        """
        Args:
            num_entities: Total number of entities in the graph.
            num_relations: Total number of relations in the graph.
            embedding_dim: Dimension of entity and relation embeddings.
            p_norm: Norm type for scoring (L1 or L2).
            init_embeddings: Whether to initialize embeddings during initialization.
        """
        super(MurE, self).__init__(num_entities, num_relations, embedding_dim, init_embeddings)
        self.p_norm = p_norm

    def _score_triplets(self, triplets):
        """
        Compute the scores for a batch of triplets.
        
        Args:
            triplets: A batch of triplets [heads, relations, tails].
        
        Returns:
            Tensor: Triplet scores.
        """
        heads, relations, tails = triplets[:, 0], triplets[:, 1], triplets[:, 2]
        head_emb, relation_emb, tail_emb = self.get_embedding(heads, relations, tails)

        # Compute the distance-based score in Euclidean space
        scores = -torch.norm(head_emb + relation_emb - tail_emb, p=self.p_norm, dim=-1)
        return scores

    def forward(self, pos_triplets, neg_triplets):
        """
        Compute the margin ranking loss for positive and negative triplets.

        Args:
            pos_triplets: Positive triplets [batch_size, 3].
            neg_triplets: Negative triplets [batch_size, 3].

        Returns:
            Loss value.
        """
        pos_scores = self._score_triplets(pos_triplets)
        neg_scores = self._score_triplets(neg_triplets)

        # Define the margin ranking loss
        target = torch.ones_like(pos_scores)
        return nn.MarginRankingLoss(margin=1.0)(pos_scores, neg_scores, target)




        
def generate_synthetic_data(num_entities, num_relations, num_triplets):
    """
    Generate positive and corrupted negative triplets.
    """
    pos_triplets = torch.randint(0, num_entities, (num_triplets, 3))
    pos_triplets[:, 1] = torch.randint(0, num_relations, (num_triplets,))  # Ensure relations are valid

    # Clone positive triplets for negative sampling
    neg_triplets = pos_triplets.clone()

    # Corrupt either head or tail
    for i in range(num_triplets):
        if random.random() < 0.5:  # Corrupt head
            neg_triplets[i, 0] = torch.randint(0, num_entities, (1,))
        else:  # Corrupt tail
            neg_triplets[i, 2] = torch.randint(0, num_entities, (1,))

    return pos_triplets, neg_triplets



def test_baselines(models, num_entities, num_relations, embedding_dim, num_triplets):
    """
    Test all baseline models using synthetic data.
    Args:
        models: Dictionary of model classes.
        num_entities: Number of unique entities.
        num_relations: Number of unique relations.
        embedding_dim: Embedding dimension size.
        num_triplets: Number of positive triplets.
    """
    # Generate synthetic data
    print("Generating synthetic data...")
    pos_triplets, neg_triplets = generate_synthetic_data(num_entities, num_relations, num_triplets)

    # Initialize and test each model
    for model_name, model_class in models.items():
        print(f"\nTesting {model_name}...")
        model = model_class(num_entities, num_relations, embedding_dim)
        model.train()

        # Compute loss
        loss = model(pos_triplets, neg_triplets)
        print(f"{model_name} Loss: {loss.item():.4f}")

if __name__ == "__main__":

    print("Models imported.")
#    # Hyperparameters
#    NUM_ENTITIES = 1000
#    NUM_RELATIONS = 100
#    EMBEDDING_DIM = 200
#    NUM_TRIPLETS = 128  # Batch size for triplets
#
#    # Dictionary of all baseline models
#    models = {
#        "TransE": TransE,
#        "TransD": TransD,
#        "TransR": TransR,
#        "DistMult": DistMult,
#        "RESCAL": RESCAL,
#        "ComplEx": ComplEx,
#        #"NTN": NTN,
#        "SimplE": SimplE,
#        "TuckER": TuckER
#    }
#
#    print("Testing Knowledge Graph Baselines...")
#    test_baselines(models, NUM_ENTITIES, NUM_RELATIONS, EMBEDDING_DIM, NUM_TRIPLETS)