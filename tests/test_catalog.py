def test_ids_unique_and_count(catalog):
    assert len(catalog.ids) == len(set(catalog.ids))
    assert len(catalog.ids) == 45


def test_crew_is_not_a_target(catalog):
    assert not any(s.startswith("crew_") for s in catalog.ids)
    assert catalog.crews == ["crew_2", "crew_3", "crew_4"]


def test_trucks_match_company_fleet(catalog):
    assert catalog.members("truck") == ["truck_10ft", "truck_16ft", "truck_26ft"]


def test_no_items_the_company_does_not_offer(catalog):
    banned = ("pool_table", "full_value_protection", "vehicle", "truck_20ft")
    assert not any(b in s for s in catalog.ids for b in banned)


def test_every_family_has_a_base_rate_and_quantity_rule(A, catalog):
    for unit in catalog.units:
        assert unit in A["unit_base_logit"], unit
        if catalog.by_id[catalog.members(unit)[0]].pricing != "storage_monthly":
            assert unit in A["quantity_rules"], unit


def test_multi_variant_families_have_weights(A, catalog):
    for unit in catalog.units:
        members = catalog.members(unit)
        if len(members) > 1:
            weights = A["variant_weights"][unit]
            for size_class, w in weights.items():
                assert set(w) <= set(members), (unit, size_class)
                assert abs(sum(w.values()) - 1) < 1e-9, (unit, size_class)


def test_only_dollies_are_free(catalog):
    assert catalog.ids_with_pricing("free") == ["furniture_dolly", "hand_truck_dolly"]


def test_every_priced_item_has_a_price(A, catalog):
    from movrec.generator.pricing import unit_prices
    prices = unit_prices(A)
    for sid in catalog.ids_with_pricing("unit"):
        assert prices[sid] > 0, sid
    for sid in catalog.ids_with_pricing("storage_monthly"):
        assert A["storage_monthly_rates"][sid] > 0, sid
