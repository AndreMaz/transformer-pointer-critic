from agents.models.transformer.actor.model import ActorTransformer
from agents.models.transformer.critic.model import CriticTransformer

def model_factory(type, opts):
    if type == 'transformer':
        selector_actor = ActorTransformer(
            opts['actor']['num_layers'],
            opts['actor']['dim_model'],
            opts['actor']['num_heads'],
            opts['actor']['inner_layer_dim'],
            opts['actor']['positional_encoding'],
            opts['vocab_size'],
            opts['actor']['logit_clipping_C'],
            opts['actor']['SOS_CODE'],
            opts['actor']['encoder_embedding_time_distributed'],
            opts['actor']['attention_dense_units'],
            opts['actor']['dropout_rate'],
            opts['actor']['use_default_initializer']
        )
        allocator_actor = ActorTransformer(
            opts['actor']['num_layers'],
            opts['actor']['dim_model'],
            opts['actor']['num_heads'],
            opts['actor']['inner_layer_dim'],
            opts['actor']['positional_encoding'],
            opts['vocab_size'],
            opts['actor']['logit_clipping_C'],
            opts['actor']['SOS_CODE'],
            opts['actor']['encoder_embedding_time_distributed'],
            opts['actor']['attention_dense_units'],
            opts['actor']['dropout_rate'],
            opts['actor']['use_default_initializer']
        )
        critic = CriticTransformer(
            opts['critic']['num_layers'],
            opts['critic']['dim_model'],
            opts['critic']['num_heads'],
            opts['critic']['inner_layer_dim'],
            opts['critic']['positional_encoding'],
            opts['vocab_size'],
            opts['critic']['encoder_embedding_time_distributed'],
            opts['critic']['last_layer_units'],
            opts['critic']['last_layer_activation'],
            opts['critic']['dropout_rate'],
            opts['critic']['use_default_initializer']
        )
    else:
        return
        

    return selector_actor, allocator_actor, critic