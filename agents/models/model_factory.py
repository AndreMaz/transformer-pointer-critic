from agents.models.transformer.actor.model import ActorTransformer
from agents.models.transformer.critic.model import CriticTransformer

def model_factory(type, opts):
    if type == 'transformer':
        allocator_actor = ActorTransformer(
            opts['actor']['num_layers'],
            opts['actor']['dim_model'],
            opts['actor']['num_heads'],
            opts['actor']['inner_layer_dim'],
            opts['actor']['logit_clipping_C'],
            opts['actor']['encoder_embedding_time_distributed'],
            opts['actor']['attention_dense_units'],
            opts['actor']['use_default_initializer'],
            opts['encoder_embedding']['common'],
            opts['encoder_embedding']['bin_features'],
            opts['encoder_embedding']['resource_features']
        )
        critic = CriticTransformer(
            opts['critic']['num_layers'],
            opts['critic']['dim_model'],
            opts['critic']['num_heads'],
            opts['critic']['inner_layer_dim'],
            opts['critic']['encoder_embedding_time_distributed'],
            opts['critic']['last_layer_units'],
            opts['critic']['last_layer_activation'],
            opts['critic']['use_default_initializer'],
            opts['encoder_embedding']['common'],
            opts['encoder_embedding']['bin_features'],
            opts['encoder_embedding']['resource_features']
        )
    else:
        return
        

    return allocator_actor, critic