"""bellwether import: corpus sets from public datasets at pinned revisions, never edited by hand."""

from __future__ import annotations

import argparse


def run(args: argparse.Namespace) -> int:
    if args.dataset == "bfcl":
        from .bfcl import run as run_bfcl

        return run_bfcl(args)
    if args.dataset == "gsm8k":
        from .gsm8k import run as run_gsm8k

        return run_gsm8k(args)
    if args.dataset == "mgsm":
        from .mgsm import run as run_mgsm

        return run_mgsm(args)
    if args.dataset == "shapes":
        from .shapes import run as run_shapes

        return run_shapes(args)
    if args.dataset == "hermes":
        from .hermes import run as run_hermes

        return run_hermes(args)
    if args.dataset == "swebench":
        from .swebench import run as run_swebench

        return run_swebench(args)
    if args.dataset == "glaive-v2":
        from .glaive_v2 import run as run_glaive_v2

        return run_glaive_v2(args)
    if args.dataset == "swehero":
        from .swehero import run as run_swehero

        return run_swehero(args)
    if args.dataset == "tau2":
        from .tau2 import run as run_tau2

        return run_tau2(args)
    if args.dataset == "mmmu":
        from .mmmu import run as run_mmmu

        return run_mmmu(args)
    if args.dataset == "docvqa":
        from .docvqa import run as run_docvqa

        return run_docvqa(args)
    if args.dataset == "textvqa":
        from .textvqa import run as run_textvqa

        return run_textvqa(args)
    if args.dataset == "coco":
        from .coco import run as run_coco

        return run_coco(args)
    if args.dataset == "chartqa":
        from .chartqa import run as run_chartqa

        return run_chartqa(args)
    if args.dataset == "scienceqa":
        from .scienceqa import run as run_scienceqa

        return run_scienceqa(args)
    if args.dataset == "ocrbench":
        from .ocrbench import run as run_ocrbench

        return run_ocrbench(args)
    if args.dataset == "librispeech":
        from .librispeech import run as run_librispeech

        return run_librispeech(args)
    if args.dataset == "mls":
        from .mls import run as run_mls

        return run_mls(args)
    if args.dataset == "voxpopuli":
        from .voxpopuli import run as run_voxpopuli

        return run_voxpopuli(args)
    if args.dataset == "fleurs":
        from .fleurs import run as run_fleurs

        return run_fleurs(args)
    if args.dataset == "peoplesspeech":
        from .peoplesspeech import run as run_peoplesspeech

        return run_peoplesspeech(args)
    if args.dataset == "gqa":
        from .gqa import run as run_gqa

        return run_gqa(args)
    if args.dataset == "wit":
        from .wit import run as run_wit

        return run_wit(args)
    if args.dataset == "doclaynet":
        from .doclaynet import run as run_doclaynet

        return run_doclaynet(args)
    if args.dataset == "pixmodocs":
        from .pixmodocs import run as run_pixmodocs

        return run_pixmodocs(args)
    if args.dataset == "websight":
        from .websight import run as run_websight

        return run_websight(args)
    if args.dataset == "docmatix":
        from .docmatix import run as run_docmatix

        return run_docmatix(args)
    if args.dataset == "mathvista":
        from .mathvista import run as run_mathvista

        return run_mathvista(args)
    if args.dataset == "cord":
        from .cord import run as run_cord

        return run_cord(args)
    if args.dataset == "pixmocap":
        from .pixmocap import run as run_pixmocap

        return run_pixmocap(args)
    if args.dataset == "vizwiz":
        from .vizwiz import run as run_vizwiz

        return run_vizwiz(args)
    if args.dataset == "infovqa":
        from .infovqa import run as run_infovqa

        return run_infovqa(args)
    if args.dataset == "pets":
        from .pets import run as run_pets

        return run_pets(args)
    if args.dataset == "realworldqa":
        from .realworldqa import run as run_realworldqa

        return run_realworldqa(args)
    if args.dataset == "vqarad":
        from .vqarad import run as run_vqarad

        return run_vqarad(args)
    if args.dataset == "pathvqa":
        from .pathvqa import run as run_pathvqa

        return run_pathvqa(args)
    if args.dataset == "mnist":
        from .mnist import run as run_mnist

        return run_mnist(args)
    if args.dataset == "invoices":
        from .invoices import run as run_invoices

        return run_invoices(args)
    raise ValueError(f"no importer for {args.dataset!r}")
