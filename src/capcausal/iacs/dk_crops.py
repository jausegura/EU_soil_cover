"""Functional classification of Danish crop codes."""
from __future__ import annotations

import pandas as pd

GRAIN_MAIZE = {5, 19}
SILAGE_MAIZE = {216, 218}
POTATOES = {149, 150, 151, 152, 153, 154, 155, 156, 157}
BEET = {160, 280}
WINTER_SOWN_SPECIAL = {70, 102, 104}
NON_AGRI = {271, 318, 321, 907}

HCAT_GRASS = {"pasture_meadow_grassland_grass", "temporary_grass", "legume_grass_mix",
              "clover_grass_mix", "alfalfa_grass_mix", "sod_turf"}
HCAT_FALLOW = {"fallow_land_not_crop"}
HCAT_FOREST = {"tree_wood_forest", "afforestation_reforestation", "christmas_trees",
               "willows_osiers", "populus", "nurseries_nursery"}

GENUS_BY_HCAT_TOKEN = [
    ("ryegrass", "lolium"), ("lolium", "lolium"), ("fescue", "festuca"), ("festuca", "festuca"),
    ("wheat", "triticum"), ("spelt", "triticum"), ("barley", "hordeum"), ("rye", "secale"),
    ("oat", "avena"), ("triticale", "triticosecale"), ("maize", "zea"), ("rapeseed", "brassicaceae"),
    ("rape", "brassicaceae"), ("radish", "brassicaceae"), ("cabbage", "brassicaceae"),
    ("cauliflower", "brassicaceae"), ("mustard", "brassicaceae"), ("potato", "solanaceae"),
    ("beet", "beta"), ("pea", "pisum"), ("bean", "vicia"), ("lupin", "lupinus"),
    ("lolium", "lolium"), ("festuca", "festuca"), ("poa", "poa"), ("timothy", "phleum"),
    ("clover", "trifolium"), ("spinach", "spinacia"), ("carrot", "daucus"), ("onion", "allium"),
    ("pumpkin", "cucurbitaceae"), ("squash", "cucurbitaceae"), ("cucumber", "cucurbitaceae"),
]


def _is_permanent_grass(code: int, name_dk: str, hcat: str) -> bool:
    name = (name_dk or "").lower()
    if hcat in HCAT_GRASS or code in {247, 254}:
        return "permanent" in name
    return False


def classify(code, name_dk: str | None, name_en: str | None, hcat: str | None) -> dict:
    """Return functional attributes for one crop code."""
    code = int(code) if pd.notna(code) else -1
    name_dk, name_en, hcat = (x if isinstance(x, str) else "" for x in (name_dk, name_en, hcat))
    en = (name_en or "").lower()
    dk = (name_dk or "").lower()

    if not hcat:
        if "brak" in dk or "udtagning" in dk and "skov" not in dk:
            hcat = "fallow_land_not_crop"
        elif "skov" in dk or "pil" in dk.split() or "poppel" in dk or "lavskov" in dk:
            hcat = "tree_wood_forest" if "skov" in dk else "willows_osiers"
        elif "kartof" in dk:
            hcat = "potatoes"
        elif "græs" in dk:
            hcat = "temporary_grass"
    perm_grass = _is_permanent_grass(code, dk, hcat)
    rot_grass = (hcat in HCAT_GRASS or code in {247}) and not perm_grass
    forest_like = hcat in HCAT_FOREST or 560 <= code <= 599
    non_agri = code in NON_AGRI or (hcat == "environmental_protection" and "ikke landbrugsareal" in dk) \
        or ("ej landbrugsareal" in dk) or hcat in {"afforestation_reforestation", "tree_wood_forest"}

    if code in GRAIN_MAIZE:
        group = "maize_grain"
    elif code in SILAGE_MAIZE:
        group = "maize_silage"
    elif code in POTATOES or hcat == "potatoes":
        group = "potatoes"
    elif code in BEET or "beet" in hcat:
        group = "beet"
    elif perm_grass:
        group = "grass_permanent"
    elif rot_grass:
        group = "grass_rotation"
    elif hcat in HCAT_FALLOW:
        group = "fallow"
    elif forest_like:
        group = "woody_or_forest"
    elif non_agri or hcat in {"environmental_protection", "not_known_and_other"}:
        group = "other_nonarable"
    elif "rapeseed" in hcat or "rape" in en:
        group = "rapeseed_winter" if ("winter" in hcat or "vinter" in dk) else "rapeseed_spring"
    elif code in {102, 104}:
        group = "seed_grass_or_herb"
    elif hcat.startswith("winter_") or "green_cut_winter" in hcat or code in WINTER_SOWN_SPECIAL:
        group = "cereal_winter"
    elif hcat.startswith("spring_") or "green_cut_spring" in hcat or hcat in {
            "cereal_legume_mix", "unspecified_cereals_cut_green", "spring_unspecified_cereals"}:
        group = "cereal_spring"
    elif hcat in {"peas", "beans", "sweet_lupins"}:
        group = "legume"
    elif any(t in hcat for t in ("lolium", "festuca", "poaceae", "timothy", "clover")) or 100 <= code <= 125:
        group = "seed_grass_or_herb"
    else:
        group = "other_arable"

    autumn_sown = group in {"cereal_winter", "rapeseed_winter"} or code in WINTER_SOWN_SPECIAL
    arable = group not in {"grass_permanent", "woody_or_forest", "other_nonarable"}
    agricultural = not non_agri

    genus = "grass_forage" if group in {"grass_rotation", "grass_permanent"} else (
        "fallow" if group == "fallow" else None)
    if genus is None:
        key = hcat + " " + en
        for token, g in GENUS_BY_HCAT_TOKEN:
            if token in key:
                genus = g
                break
    if genus is None:
        genus = hcat or f"code{code}"
    season = "w" if autumn_sown else ("s" if group in {"cereal_spring", "rapeseed_spring"} else "")
    glm7_category = f"{genus}_{season}" if season else genus

    return {"crop_group": group, "autumn_sown": autumn_sown, "arable": arable,
            "agricultural": agricultural, "glm7_category": glm7_category}


def classify_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Classify unique (crop_code, crop_name, EC_trans_n, EC_hcat_n) rows and merge back."""
    keys = ["crop_code", "crop_name", "EC_trans_n", "EC_hcat_n"]
    u = df[keys].drop_duplicates().reset_index(drop=True)
    attrs = pd.DataFrame([classify(r.crop_code, r.crop_name, r.EC_trans_n, r.EC_hcat_n)
                          for r in u.itertuples(index=False)])
    return df.merge(pd.concat([u, attrs], axis=1), on=keys, how="left")
