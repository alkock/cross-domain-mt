from .generator import Generator
from .discriminator import Discriminator
from ..registry import register_model


@register_model("pix2pix")
def build_pix2pix(config):
    gen = Generator(**config["generator"])
    disc = Discriminator(**config["discriminator"])
    return gen, disc
