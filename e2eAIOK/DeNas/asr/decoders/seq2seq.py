import torch

from asr.decoders.ctc import CTCPrefixScorer


class S2SBaseSearcher(torch.nn.Module):
    """S2SBaseSearcher class to be inherited by other
    decoding approaches for seq2seq model.

    Arguments
    ---------
    bos_index : int
        The index of the beginning-of-sequence (bos) token.
    eos_index : int
        The index of end-of-sequence token.
    min_decode_radio : float
        The ratio of minimum decoding steps to the length of encoder states.
    max_decode_radio : float
        The ratio of maximum decoding steps to the length of encoder states.

    Returns
    -------
    predictions
        Outputs as Python list of lists, with "ragged" dimensions; padding
        has been removed.
    scores
        The sum of log probabilities (and possibly
        additional heuristic scores) for each prediction.

    """

    def __init__(
        self, bos_index, eos_index, min_decode_ratio, max_decode_ratio,
    ):
        super(S2SBaseSearcher, self).__init__()
        self.bos_index = bos_index
        self.eos_index = eos_index
        self.min_decode_ratio = min_decode_ratio
        self.max_decode_ratio = max_decode_ratio

    def forward(self, enc_states, wav_len):
        """This method should implement the forward algorithm of decoding method.

        Arguments
        ---------
        enc_states : torch.Tensor
            The precomputed encoder states to be used when decoding.
            (ex. the encoded speech representation to be attended).
        wav_len : torch.Tensor
            relative length.
        """
        raise NotImplementedError

    def forward_step(self, inp_tokens, memory, enc_states, enc_lens):
        """This method should implement one step of
        forwarding operation in the autoregressive model.

        Arguments
        ---------
        inp_tokens : torch.Tensor
            The input tensor of the current timestep.
        memory : No limit
            The memory variables input for this timestep.
            (ex. RNN hidden states).
        enc_states : torch.Tensor
            The encoder states to be attended.
        enc_lens : torch.Tensor
            The actual length of each enc_states sequence.

        Returns
        -------
        log_probs : torch.Tensor
            Log-probabilities of the current timestep output.
        memory : No limit
            The memory variables generated in this timestep.
            (ex. RNN hidden states).
        attn : torch.Tensor
            The attention weight for doing penalty.
        """
        raise NotImplementedError

    def reset_mem(self, batch_size, device):
        """This method should implement the resetting of
        memory variables for the seq2seq model.
        E.g., initializing zero vector as initial hidden states.

        Arguments
        ---------
        batch_size : int
            The size of the batch.
        device : torch.device
            The device to put the initial variables.

        Return
        ------
        memory : No limit
            The initial memory variable.
        """
        raise NotImplementedError

    def lm_forward_step(self, inp_tokens, memory):
        """This method should implement one step of
        forwarding operation for language model.

        Arguments
        ---------
        inp_tokens : torch.Tensor
            The input tensor of the current timestep.
        memory : No limit
            The momory variables input for this timestep.
            (e.g., RNN hidden states).

        Return
        ------
        log_probs : torch.Tensor
            Log-probabilities of the current timestep output.
        memory : No limit
            The memory variables generated in this timestep.
            (e.g., RNN hidden states).
        """
        raise NotImplementedError

    def reset_lm_mem(self, batch_size, device):
        """This method should implement the resetting of
        memory variables in the language model.
        E.g., initializing zero vector as initial hidden states.

        Arguments
        ---------
        batch_size : int
            The size of the batch.
        device : torch.device
            The device to put the initial variables.

        Return
        ------
        memory : No limit
            The initial memory variable.
        """
        raise NotImplementedError


class S2SBeamSearcher(S2SBaseSearcher):
    """This class implements the beam-search algorithm for the seq2seq model.
    See also S2SBaseSearcher().

    Arguments
    ---------
    bos_index : int
        The index of beginning-of-sequence token.
    eos_index : int
        The index of end-of-sequence token.
    min_decode_radio : float
        The ratio of minimum decoding steps to length of encoder states.
    max_decode_radio : float
        The ratio of maximum decoding steps to length of encoder states.
    beam_size : int
        The width of beam.
    topk : int
        The number of hypothesis to return. (default: 1)
    return_log_probs : bool
        Whether to return log-probabilities. (default: False)
    using_eos_threshold : bool
        Whether to use eos threshold. (default: true)
    eos_threshold : float
        The threshold coefficient for eos token (default: 1.5). See 3.1.2 in
        reference: https://arxiv.org/abs/1904.02619
    length_normalization : bool
        Whether to divide the scores by the length. (default: True)
    length_rewarding : float
        The coefficient of length rewarding (γ).
        log P(y|x) + λ log P_LM(y) + γ*len(y). (default: 0.0)
    coverage_penalty: float
        The coefficient of coverage penalty (η).
        log P(y|x) + λ log P_LM(y) + γ*len(y) + η*coverage(x,y). (default: 0.0)
        Reference: https://arxiv.org/pdf/1612.02695.pdf, https://arxiv.org/pdf/1808.10792.pdf
    lm_weight : float
        The weight of LM when performing beam search (λ).
        log P(y|x) + λ log P_LM(y). (default: 0.0)
    ctc_weight : float
        The weight of CTC probabilities when performing beam search (λ).
        (1-λ) log P(y|x) + λ log P_CTC(y|x). (default: 0.0)
    blank_index : int
        The index of the blank token.
    ctc_score_mode: str
        Default: "full"
        CTC prefix scoring on "partial" token or "full: token.
    ctc_window_size: int
        Default: 0
        Compute the ctc scores over the time frames using windowing based on attention peaks.
        If 0, no windowing applied.
    using_max_attn_shift: bool
        Whether using the max_attn_shift constraint. (default: False)
    max_attn_shift: int
        Beam search will block the beams that attention shift more
        than max_attn_shift.
        Reference: https://arxiv.org/abs/1904.02619
    minus_inf : float
        DefaultL -1e20
        The value of minus infinity to block some path
        of the search.
    """

    def __init__(
        self,
        bos_index,
        eos_index,
        min_decode_ratio,
        max_decode_ratio,
        beam_size,
        topk=1,
        return_log_probs=False,
        using_eos_threshold=True,
        eos_threshold=1.5,
        length_normalization=True,
        length_rewarding=0,
        coverage_penalty=0.0,
        lm_weight=0.0,
        lm_modules=None,
        ctc_weight=0.0,
        blank_index=0,
        ctc_score_mode="full",
        ctc_window_size=0,
        using_max_attn_shift=False,
        max_attn_shift=60,
        minus_inf=-1e20,
    ):
        super(S2SBeamSearcher, self).__init__(
            bos_index, eos_index, min_decode_ratio, max_decode_ratio,
        )
        self.beam_size = beam_size
        self.topk = topk
        self.return_log_probs = return_log_probs
        self.length_normalization = length_normalization
        self.length_rewarding = length_rewarding
        self.coverage_penalty = coverage_penalty
        self.coverage = None

        if self.length_normalization and self.length_rewarding > 0:
            raise ValueError(
                "length normalization is not compatible with length rewarding."
            )

        self.using_eos_threshold = using_eos_threshold
        self.eos_threshold = eos_threshold
        self.using_max_attn_shift = using_max_attn_shift
        self.max_attn_shift = max_attn_shift
        self.lm_weight = lm_weight
        self.lm_modules = lm_modules

        # ctc related
        self.ctc_weight = ctc_weight
        self.blank_index = blank_index
        self.att_weight = 1.0 - ctc_weight

        assert (
            0.0 <= self.ctc_weight <= 1.0
        ), "ctc_weight should not > 1.0 and < 0.0"

        if self.ctc_weight > 0.0:
            if len({self.bos_index, self.eos_index, self.blank_index}) < 3:
                raise ValueError(
                    "To perform joint ATT/CTC decoding, set blank, eos and bos to different indexes."
                )

        # ctc already initialized
        self.minus_inf = minus_inf
        self.ctc_score_mode = ctc_score_mode
        self.ctc_window_size = ctc_window_size

    def _check_full_beams(self, hyps, beam_size):
        """This method checks whether hyps has been full.

        Arguments
        ---------
        hyps : List
            This list contains batch_size number.
            Each inside list contains a list stores all the hypothesis for this sentence.
        beam_size : int
            The number of beam_size.

        Returns
        -------
        bool
            Whether the hyps has been full.
        """
        hyps_len = [len(lst) for lst in hyps]
        beam_size = [self.beam_size for _ in range(len(hyps_len))]
        if hyps_len == beam_size:
            return True
        else:
            return False

    def _check_attn_shift(self, attn, prev_attn_peak):
        """This method checks whether attention shift is more than attn_shift.

        Arguments
        ---------
        attn : torch.Tensor
            The attention to be checked.
        prev_attn_peak : torch.Tensor
            The previous attention peak place.

        Returns
        -------
        cond : torch.BoolTensor
            Each element represents whether the beam is within the max_shift range.
        attn_peak : torch.Tensor
            The peak of the attn tensor.
        """
        # Block the candidates that exceed the max shift
        _, attn_peak = torch.max(attn, dim=1)
        lt_cond = attn_peak <= (prev_attn_peak + self.max_attn_shift)
        mt_cond = attn_peak > (prev_attn_peak - self.max_attn_shift)

        # True if not exceed limit
        # Multiplication equals to element-wise and for tensor
        cond = (lt_cond * mt_cond).unsqueeze(1)
        return cond, attn_peak

    def _check_eos_threshold(self, log_probs):
        """
        This method checks whether eos log-probabilities exceed threshold.

        Arguments
        ---------
        log_probs : torch.Tensor
            The log-probabilities.

        Return
        ------
        cond : torch.BoolTensor
            Each element represents whether the eos log-probabilities will be kept.
        """
        max_probs, _ = torch.max(log_probs, dim=-1)
        eos_probs = log_probs[:, self.eos_index]
        cond = eos_probs > (self.eos_threshold * max_probs)
        return cond

    def _update_hyp_and_scores(
        self,
        inp_tokens,
        alived_seq,
        alived_log_probs,
        hyps_and_scores,
        scores,
        timesteps,
    ):
        """This method will update hyps and scores if inp_tokens are eos.

        Arguments
        ---------
        inp_tokens : torch.Tensor
            The current output.
        alived_seq : torch.Tensor
            The tensor to store the alived_seq.
        alived_log_probs : torch.Tensor
            The tensor to store the alived_log_probs.
        hyps_and_scores : list
            To store generated hypotheses and scores.
        scores : torch.Tensor
            The final scores of beam search.
        timesteps : float
            The current timesteps. This is for length rewarding.

        Returns
        -------
        is_eos : torch.BoolTensor
            Each element represents whether the token is eos.
        """
        is_eos = inp_tokens.eq(self.eos_index)
        (eos_indices,) = torch.nonzero(is_eos, as_tuple=True)

        # Store the hypothesis and their scores when reaching eos.
        if eos_indices.shape[0] > 0:
            for index in eos_indices:
                # convert to int
                index = index.item()
                batch_id = torch.div(
                    index, self.beam_size, rounding_mode="floor"
                )
                if len(hyps_and_scores[batch_id]) == self.beam_size:
                    continue
                hyp = alived_seq[index, :]
                log_probs = alived_log_probs[index, :]
                final_scores = scores[index] + self.length_rewarding * (
                    timesteps + 1
                )
                hyps_and_scores[batch_id].append((hyp, log_probs, final_scores))
        return is_eos

    def _get_top_score_prediction(self, hyps_and_scores, topk):
        """This method sorts the scores and return corresponding hypothesis and log probs.

        Arguments
        ---------
        hyps_and_scores : list
            To store generated hypotheses and scores.
        topk : int
            Number of hypothesis to return.

        Returns
        -------
        topk_hyps : torch.Tensor (batch, topk, max length of token_id sequences)
            This tensor stores the topk predicted hypothesis.
        topk_scores : torch.Tensor (batch, topk)
            The length of each topk sequence in the batch.
        topk_lengths : torch.Tensor (batch, topk)
            This tensor contains the final scores of topk hypotheses.
        topk_log_probs : list
            The log probabilities of each hypotheses.
        """
        top_hyps, top_log_probs, top_scores, top_lengths = [], [], [], []
        batch_size = len(hyps_and_scores)

        # Collect hypotheses
        for i in range(len(hyps_and_scores)):
            hyps, log_probs, scores = zip(*hyps_and_scores[i])
            top_hyps += hyps
            top_scores += scores
            top_log_probs += log_probs
            top_lengths += [len(hyp) for hyp in hyps]
        top_hyps = torch.nn.utils.rnn.pad_sequence(
            top_hyps, batch_first=True, padding_value=0
        )
        top_scores = torch.stack((top_scores), dim=0).view(batch_size, -1)
        top_lengths = torch.tensor(
            top_lengths, dtype=torch.int, device=top_scores.device
        )
        # Get topk indices
        topk_scores, indices = top_scores.topk(self.topk, dim=-1)
        indices = (indices + self.beam_offset.unsqueeze(1)).view(
            batch_size * self.topk
        )
        # Select topk hypotheses
        topk_hyps = torch.index_select(top_hyps, dim=0, index=indices,)
        topk_hyps = topk_hyps.view(batch_size, self.topk, -1)
        topk_lengths = torch.index_select(top_lengths, dim=0, index=indices,)
        topk_lengths = topk_lengths.view(batch_size, self.topk)
        topk_log_probs = [top_log_probs[index.item()] for index in indices]

        return topk_hyps, topk_scores, topk_lengths, topk_log_probs

    def forward(self, enc_states, wav_len):  # noqa: C901
        """Applies beamsearch and returns the predicted tokens."""
        enc_lens = torch.round(enc_states.shape[1] * wav_len).int()
        device = enc_states.device
        batch_size = enc_states.shape[0]

        memory = self.reset_mem(batch_size * self.beam_size, device=device)

        if self.lm_weight > 0:
            lm_memory = self.reset_lm_mem(batch_size * self.beam_size, device)

        if self.ctc_weight > 0:
            # (batch_size * beam_size, L, vocab_size)
            ctc_outputs = self.ctc_forward_step(enc_states)
            ctc_scorer = CTCPrefixScorer(
                ctc_outputs,
                enc_lens,
                batch_size,
                self.beam_size,
                self.blank_index,
                self.eos_index,
                self.ctc_window_size,
            )
            ctc_memory = None

        # Inflate the enc_states and enc_len by beam_size times
        enc_states = inflate_tensor(enc_states, times=self.beam_size, dim=0)
        enc_lens = inflate_tensor(enc_lens, times=self.beam_size, dim=0)

        # Using bos as the first input
        inp_tokens = (
            torch.zeros(batch_size * self.beam_size, device=device)
            .fill_(self.bos_index)
            .long()
        )

        # The first index of each sentence.
        self.beam_offset = (
            torch.arange(batch_size, device=device) * self.beam_size
        )

        # initialize sequence scores variables.
        sequence_scores = torch.empty(
            batch_size * self.beam_size, device=device
        )
        sequence_scores.fill_(float("-inf"))

        # keep only the first to make sure no redundancy.
        sequence_scores.index_fill_(0, self.beam_offset, 0.0)

        # keep the hypothesis that reaches eos and their corresponding score and log_probs.
        hyps_and_scores = [[] for _ in range(batch_size)]

        # keep the sequences that still not reaches eos.
        alived_seq = torch.empty(
            batch_size * self.beam_size, 0, device=device
        ).long()

        # Keep the log-probabilities of alived sequences.
        alived_log_probs = torch.empty(
            batch_size * self.beam_size, 0, device=device
        )

        min_decode_steps = int(enc_states.shape[1] * self.min_decode_ratio)
        max_decode_steps = int(enc_states.shape[1] * self.max_decode_ratio)

        # Initialize the previous attention peak to zero
        # This variable will be used when using_max_attn_shift=True
        prev_attn_peak = torch.zeros(batch_size * self.beam_size, device=device)

        for t in range(max_decode_steps):
            # terminate condition
            if self._check_full_beams(hyps_and_scores, self.beam_size):
                break

            log_probs, memory, attn = self.forward_step(
                inp_tokens, memory, enc_states, enc_lens
            )
            log_probs = self.att_weight * log_probs

            # Keep the original value
            log_probs_clone = log_probs.clone().reshape(batch_size, -1)
            vocab_size = log_probs.shape[-1]

            if self.using_max_attn_shift:
                # Block the candidates that exceed the max shift
                cond, attn_peak = self._check_attn_shift(attn, prev_attn_peak)
                log_probs = mask_by_condition(
                    log_probs, cond, fill_value=self.minus_inf
                )
                prev_attn_peak = attn_peak

            # Set eos to minus_inf when less than minimum steps.
            if t < min_decode_steps:
                log_probs[:, self.eos_index] = self.minus_inf

            # Set the eos prob to minus_inf when it doesn't exceed threshold.
            if self.using_eos_threshold:
                cond = self._check_eos_threshold(log_probs)
                log_probs[:, self.eos_index] = mask_by_condition(
                    log_probs[:, self.eos_index],
                    cond,
                    fill_value=self.minus_inf,
                )

            # adding LM scores to log_prob if lm_weight > 0
            if self.lm_weight > 0:
                lm_log_probs, lm_memory = self.lm_forward_step(
                    inp_tokens, lm_memory
                )
                log_probs = log_probs + self.lm_weight * lm_log_probs

            # adding CTC scores to log_prob if ctc_weight > 0
            if self.ctc_weight > 0:
                g = alived_seq
                # block blank token
                log_probs[:, self.blank_index] = self.minus_inf
                if self.ctc_weight != 1.0 and self.ctc_score_mode == "partial":
                    # pruning vocab for ctc_scorer
                    _, ctc_candidates = log_probs.topk(
                        self.beam_size * 2, dim=-1
                    )
                else:
                    ctc_candidates = None

                ctc_log_probs, ctc_memory = ctc_scorer.forward_step(
                    g, ctc_memory, ctc_candidates, attn
                )
                log_probs = log_probs + self.ctc_weight * ctc_log_probs

            scores = sequence_scores.unsqueeze(1).expand(-1, vocab_size)
            scores = scores + log_probs

            # length normalization
            if self.length_normalization:
                scores = scores / (t + 1)

            # keep topk beams
            scores, candidates = scores.view(batch_size, -1).topk(
                self.beam_size, dim=-1
            )

            # The input for the next step, also the output of current step.
            inp_tokens = (candidates % vocab_size).view(
                batch_size * self.beam_size
            )

            scores = scores.view(batch_size * self.beam_size)
            sequence_scores = scores

            # recover the length normalization
            if self.length_normalization:
                sequence_scores = sequence_scores * (t + 1)

            # The index of which beam the current top-K output came from in (t-1) timesteps.
            predecessors = (
                torch.div(candidates, vocab_size, rounding_mode="floor")
                + self.beam_offset.unsqueeze(1).expand_as(candidates)
            ).view(batch_size * self.beam_size)

            # Permute the memory to synchoronize with the output.
            memory = self.permute_mem(memory, index=predecessors)
            if self.lm_weight > 0:
                lm_memory = self.permute_lm_mem(lm_memory, index=predecessors)

            if self.ctc_weight > 0:
                ctc_memory = ctc_scorer.permute_mem(ctc_memory, candidates)

            # If using_max_attn_shift, then the previous attn peak has to be permuted too.
            if self.using_max_attn_shift:
                prev_attn_peak = torch.index_select(
                    prev_attn_peak, dim=0, index=predecessors
                )

            # Add coverage penalty
            if self.coverage_penalty > 0:
                cur_attn = torch.index_select(attn, dim=0, index=predecessors)

                # coverage: cumulative attention probability vector
                if t == 0:
                    # Init coverage
                    self.coverage = cur_attn

                # the attn of transformer is [batch_size*beam_size, current_step, source_len]
                if len(cur_attn.size()) > 2:
                    self.converage = torch.sum(cur_attn, dim=1)
                else:
                    # Update coverage
                    self.coverage = torch.index_select(
                        self.coverage, dim=0, index=predecessors
                    )
                    self.coverage = self.coverage + cur_attn

                # Compute coverage penalty and add it to scores
                penalty = torch.max(
                    self.coverage, self.coverage.clone().fill_(0.5)
                ).sum(-1)
                penalty = penalty - self.coverage.size(-1) * 0.5
                penalty = penalty.view(batch_size * self.beam_size)
                penalty = (
                    penalty / (t + 1) if self.length_normalization else penalty
                )
                scores = scores - penalty * self.coverage_penalty

            # Update alived_seq
            alived_seq = torch.cat(
                [
                    torch.index_select(alived_seq, dim=0, index=predecessors),
                    inp_tokens.unsqueeze(1),
                ],
                dim=-1,
            )

            # Takes the log-probabilities
            beam_log_probs = log_probs_clone[
                torch.arange(batch_size).unsqueeze(1), candidates
            ].reshape(batch_size * self.beam_size)
            alived_log_probs = torch.cat(
                [
                    torch.index_select(
                        alived_log_probs, dim=0, index=predecessors
                    ),
                    beam_log_probs.unsqueeze(1),
                ],
                dim=-1,
            )

            is_eos = self._update_hyp_and_scores(
                inp_tokens,
                alived_seq,
                alived_log_probs,
                hyps_and_scores,
                scores,
                timesteps=t,
            )

            # Block the paths that have reached eos.
            sequence_scores.masked_fill_(is_eos, float("-inf"))

        if not self._check_full_beams(hyps_and_scores, self.beam_size):
            # Using all eos to fill-up the hyps.
            eos = (
                torch.zeros(batch_size * self.beam_size, device=device)
                .fill_(self.eos_index)
                .long()
            )
            _ = self._update_hyp_and_scores(
                eos,
                alived_seq,
                alived_log_probs,
                hyps_and_scores,
                scores,
                timesteps=max_decode_steps,
            )

        (
            topk_hyps,
            topk_scores,
            topk_lengths,
            log_probs,
        ) = self._get_top_score_prediction(hyps_and_scores, topk=self.topk,)
        # pick the best hyp
        predictions = topk_hyps[:, 0, :]
        predictions = batch_filter_seq2seq_output(
            predictions, eos_id=self.eos_index
        )

        if self.return_log_probs:
            return predictions, topk_scores, log_probs
        else:
            return predictions, topk_scores

    def ctc_forward_step(self, x):
        """Applies a ctc step during bramsearch."""
        logits = self.ctc_fc(x)
        log_probs = self.softmax(logits)
        return log_probs

    def permute_mem(self, memory, index):
        """This method permutes the seq2seq model memory
        to synchronize the memory index with the current output.

        Arguments
        ---------
        memory : No limit
            The memory variable to be permuted.
        index : torch.Tensor
            The index of the previous path.

        Return
        ------
        The variable of the memory being permuted.

        """
        raise NotImplementedError

    def permute_lm_mem(self, memory, index):
        """This method permutes the language model memory
        to synchronize the memory index with the current output.

        Arguments
        ---------
        memory : No limit
            The memory variable to be permuted.
        index : torch.Tensor
            The index of the previous path.

        Returns
        -------
        The variable of the memory being permuted.
        """
        raise NotImplementedError


def inflate_tensor(tensor, times, dim):
    """This function inflates the tensor for times along dim.

    Arguments
    ---------
    tensor : torch.Tensor
        The tensor to be inflated.
    times : int
        The tensor will inflate for this number of times.
    dim : int
        The dim to be inflated.

    Returns
    -------
    torch.Tensor
        The inflated tensor.
    """
    return torch.repeat_interleave(tensor, times, dim=dim)


def mask_by_condition(tensor, cond, fill_value):
    """This function will mask some element in the tensor with fill_value, if condition=False.

    Arguments
    ---------
    tensor : torch.Tensor
        The tensor to be masked.
    cond : torch.BoolTensor
        This tensor has to be the same size as tensor.
        Each element represents whether to keep the value in tensor.
    fill_value : float
        The value to fill in the masked element.

    Returns
    -------
    torch.Tensor
        The masked tensor.
    """
    tensor = torch.where(
        cond, tensor, torch.Tensor([fill_value]).to(tensor.device)
    )
    return tensor


def _update_mem(inp_tokens, memory):
    """This function is for updating the memory for transformer searches.
    it is called at each decoding step. When being called, it appends the
    predicted token of the previous step to existing memory.

    Arguments:
    -----------
    inp_tokens : tensor
        Predicted token of the previous decoding step.
    memory : tensor
        Contains all the predicted tokens.
    """
    if memory is None:
        return inp_tokens.unsqueeze(1)
    return torch.cat([memory, inp_tokens.unsqueeze(1)], dim=-1)


class S2STransformerBeamSearch(S2SBeamSearcher):
    """This class implements the beam search decoding
    for Transformer.
    See also S2SBaseSearcher(), S2SBeamSearcher().

    Arguments
    ---------
    model : torch.nn.Module
        The model to use for decoding.
    linear : torch.nn.Module
        A linear output layer.
    **kwargs
        Arguments to pass to S2SBeamSearcher
    """

    def __init__(
        self, modules, temperature=1.0, temperature_lm=1.0, **kwargs,
    ):
        super(S2STransformerBeamSearch, self).__init__(**kwargs)

        self.model = modules[0]
        self.fc = modules[1]
        self.ctc_fc = modules[2]
        self.softmax = torch.nn.LogSoftmax(dim=-1)

        self.temperature = temperature
        self.temperature_lm = temperature_lm

    def reset_mem(self, batch_size, device):
        """Needed to reset the memory during beamsearch."""
        return None

    def reset_lm_mem(self, batch_size, device):
        """Needed to reset the LM memory during beamsearch."""
        return None

    def permute_mem(self, memory, index):
        """Permutes the memory."""
        memory = torch.index_select(memory, dim=0, index=index)
        return memory

    def permute_lm_mem(self, memory, index):
        """Permutes the memory of the language model."""
        memory = torch.index_select(memory, dim=0, index=index)
        return memory

    def forward_step(self, inp_tokens, memory, enc_states, enc_lens):
        """Performs a step in the implemented beamsearcher."""
        memory = _update_mem(inp_tokens, memory)
        pred, attn = self.model.decode(memory, enc_states)
        prob_dist = self.softmax(self.fc(pred) / self.temperature)
        return prob_dist[:, -1, :], memory, attn

    def lm_forward_step(self, inp_tokens, memory):
        """Performs a step in the implemented LM module."""
        memory = _update_mem(inp_tokens, memory)
        if not next(self.lm_modules.parameters()).is_cuda:
            self.lm_modules.to(inp_tokens.device)
        logits = self.lm_modules(memory)
        log_probs = self.softmax(logits / self.temperature_lm)
        return log_probs[:, -1, :], memory


def batch_filter_seq2seq_output(prediction, eos_id=-1):
    """Calling batch_size times of filter_seq2seq_output.

    Arguments
    ---------
    prediction : list of torch.Tensor
        A list containing the output ints predicted by the seq2seq system.
    eos_id : int, string
        The id of the eos.

    Returns
    ------
    list
        The output predicted by seq2seq model.
    """
    outputs = []
    for p in prediction:
        res = filter_seq2seq_output(p.tolist(), eos_id=eos_id)
        outputs.append(res)
    return outputs


def filter_seq2seq_output(string_pred, eos_id=-1):
    """Filter the output until the first eos occurs (exclusive).

    Arguments
    ---------
    string_pred : list
        A list containing the output strings/ints predicted by the seq2seq system.
    eos_id : int, string
        The id of the eos.

    Returns
    ------
    list
        The output predicted by seq2seq model.
    """
    if isinstance(string_pred, list):
        try:
            eos_index = next(
                i for i, v in enumerate(string_pred) if v == eos_id
            )
        except StopIteration:
            eos_index = len(string_pred)
        string_out = string_pred[:eos_index]
    else:
        raise ValueError("The input must be a list.")
    return string_out
