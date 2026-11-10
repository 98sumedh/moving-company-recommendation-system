# Assumptions

Generated from `config/assumptions.yaml` (the source of truth). All data in this repo is **synthetic**.

| Key | Description | Source type | Confidence | Source |
|---|---|---|---|---|
| `n_bookings` | Bookings over the 2-year window (about 68 per day) | company_kb | high | Company staff, Nov 2026: the company does about 70 jobs per day |
| `date_range` | Move-date window. Two full peak seasons; ends before the present. | generic_reasoning | high | Chosen so a later season can be held out for testing |
| `annual_growth` | Year-over-year volume growth (linear over the window) | generic_reasoning | low | Modest growth for an established regional mover |
| `month_multiplier` | Relative move volume by month. Gives ~42% of moves in May-Aug and ~51% in May-Sep. | public_derived | medium | MoveBuddha: ~41% of US moves happen May-Aug; National Van Lines: roughly half happen May-Sep. Winter dip kept mild for San Diego weather. (https://movebuddha.com/blog/peak-moving-season/ ; https://www.nationalvanlines.com/blog/why-moving-june-costs-22-more-how-time-right/) |
| `weekday_multiplier` | Relative volume by day of week | generic_reasoning | medium | Fridays and Saturdays are popular because customers are off work; Sunday is lighter |
| `month_end_lift` | Extra volume around month end and the 1st (leases turn over) | public_derived | medium | MoveBuddha notes June 1 and June 30 as the busiest moving dates (https://movebuddha.com/blog/peak-moving-season/) |
| `holiday_multiplier` | Few moves on major holidays | generic_reasoning | medium | Crews and customers rarely move on these days |
| `move_type_probs` | Share of bookings by move type. The type list comes from the KB; the shares are guesses. | generic_reasoning | low | Types: company KB (residential/commercial, local/long-distance, labor-only, single items). Shares: local-heavy mover. |
| `home_size_probs` | Home or office size given move type | generic_reasoning | low | Apartment-heavy coastal metro; labor-only customers skew smaller |
| `sq_ft_median` | Median square feet by size (lognormal around it; 0 = not applicable) | generic_reasoning | medium | Typical US unit sizes |
| `sq_ft_sigma` | Lognormal spread of square feet | generic_reasoning | low | Spread so the largest 4-bedrooms pass the 2,000 sq ft walkthrough threshold |
| `weight_median_lbs` | Median shipment weight by size (lognormal around it) | public | medium | National Van Lines: 1BR 2,000-3,000 lb, 2BR 4,000-6,000 lb; SafeBound citing AMSA: 3BR ~7,100 lb; Movers.com: ~1,000 lb per room. Offices are guesses. (https://www.nationalvanlines.com/blog/how-do-long-distance-movers-calculate-weight-and-why-does-it-matter-for-your-budget/ ; https://www.safeboundmoving.com/blog/moving-costs-by-household-size-in-2026 ; https://www.movers.com/ask-moving-expert/what-is-the-average-weight-of-a-four-4-bedroom-house-23395) |
| `weight_sigma` | Lognormal spread of weight | generic_reasoning | medium | Public guides give ranges about +/-25% around the midpoint |
| `crew_by_home_size` | Typical crew by size. Residential from the KB price table; offices are guesses. | company_kb | high | Company KB, section 8 (typical crew by home size) |
| `crew_noise_prob` | Chance the office picks one crew size up or down from the typical one | generic_reasoning | low | KB: the office confirms the right crew for each job, so it varies |
| `truck_probs` | Truck size given home size (company fleet is 10/16/26 ft; labor-only has none) | public_derived | medium | U-Haul guide: 10 ft ~ 1 bedroom, 15 ft ~ 2, 20 ft ~ 2-3, 26 ft ~ 3-4. KB: 16 ft for studio/1BR, 26 ft for 4+. Mapped onto the company's 10/16/26 ft fleet. (https://www.uhaul.com/Tips/Loading/How-To-Choose-The-Right-Size-Moving-Truck-Rental-15591) |
| `service_area` | Where the company works | company_kb | high | Company KB, section 1 (service area) |
| `local_regions` | Local regions and their share of moves | generic_reasoning | low | San Diego-based company; most jobs in SD County |
| `local_same_region_prob` | Chance a local move stays in the same region | generic_reasoning | low | Most local moves are short |
| `local_drive_minutes` | Median drive time between pickup and drop-off, and miles per minute of driving | generic_reasoning | low | Rough SD/OC freeway driving times |
| `long_distance_destinations` | Long-distance endpoints (approximate road miles from San Diego) and their share | generic_reasoning | medium | Approximate road distances from public maps; shares are guesses |
| `long_distance_miles_sigma` | Lognormal spread of exact long-distance miles around the city distance (capped at the service-area limit) | generic_reasoning | medium | Addresses vary within a metro area |
| `long_distance_outbound_prob` | Share of long-distance moves leaving the local area (rest arrive into it) | generic_reasoning | low | Roughly balanced flows |
| `long_distance_mph` | Average truck speed for drive time on long moves | generic_reasoning | medium | Loaded truck on freeways with stops |
| `season_months` | Months that count as summer and as the rainy season (used by add-on rules) | generic_reasoning | medium | San Diego gets most of its rain November-March |
| `apartment_prob` | Chance an address is in a multi-floor building (apartment or office building) | generic_reasoning | low | Smaller homes are mostly apartments |
| `apartment_floor_probs` | Floor of an apartment or office address | generic_reasoning | low | San Diego housing is mostly low-rise |
| `elevator_prob_by_floor` | Chance the building has a usable elevator, by floor | generic_reasoning | low | Higher floors more likely to have an elevator |
| `two_story_house_prob` | Chance a house has an internal staircase (counts as 1 flight) | generic_reasoning | low | Larger houses are more often two-story |
| `payment_method_probs` | Payment method. The method list comes from the KB; the shares are guesses. | generic_reasoning | low | KB section 6 lists cash/check, Venmo, card |
| `walkthrough_sq_ft` | Homes over this size get a free in-person walkthrough | company_kb | high | Company KB, section 5 (estimates) |
| `walkthrough_extra_prob` | Chance a smaller job still gets an in-person walkthrough | generic_reasoning | low | Some customers ask for one anyway |
| `lead_time_median_days` | Median days between booking and move (lognormal) | company_kb | medium | KB: book 2-3 weeks ahead if possible; same/next-day sometimes available. Per-type values are guesses around that. |
| `lead_time_sigma` | Lognormal spread of lead time | generic_reasoning | low | Wide spread: some book months ahead, some days ahead |
| `lead_time_max_days` | Longest lead time | generic_reasoning | low | Bookings rarely open more than ~4 months out |
| `same_next_day_share` | Share of non-long-distance bookings made for the same or next day | generic_reasoning | low | KB says same/next-day may be available; share is a guess |
| `segment_lead_time_multiplier` | Lead time multipliers for some hidden segments (others 1.0) | generic_reasoning | low | Military orders and downsizing are planned early; students book late |
| `office_hours` | Hours when bookings are confirmed (booking timestamps fall inside them) | company_kb | high | Company KB, section 1 (office hours) |
| `arrival_window_start_hour` | Start of the crew arrival window | company_kb | high | Company KB, section 10 (8-9 AM and 1-3 PM windows) |
| `afternoon_share` | Share of jobs in the afternoon window | generic_reasoning | low | Most moves start in the morning |
| `status_probs` | Final booking status (rescheduled = moved to another date with no fee; move_date is the final date) | generic_reasoning | low | KB allows free rescheduling and cancellation; shares are guesses |
| `late_cancel_share` | Share of cancellations made within 48 hours (deposit forfeited) | generic_reasoning | low | KB: cancelling within 48 hours forfeits the deposit; share is a guess |
| `deposit` | Deposit to reserve a move (counts toward the total) | company_kb | high | Company KB, section 5 (booking) |
| `repeat_prob` | Chance a booking belongs to an existing customer. Most customers book once. | generic_reasoning | low | Residential moves are rare events; office accounts come back |
| `repeat_storage_pool_prob` | When a booking is a repeat, chance it comes from a customer who had storage (moving out of storage) | generic_reasoning | low | Storage customers come back to move their things out |
| `repeat_min_gap_days` | Minimum days between a customer's moves | generic_reasoning | low | Back-to-back moves within a month are rare |
| `segment_base_logit` | Base log-odds of each non-commercial hidden segment (softmax). Never exported to modeling files. | generic_reasoning | low | San Diego has large Navy/USMC and university populations; collectors are rare |
| `commercial_segment_probs` | Hidden segment for commercial/office bookings | generic_reasoning | low | Most office moves behave alike; some are lean, some have art/equipment |
| `segment_profile_effects` | Soft links from observable profile to hidden segment (so the profile only partly reveals it) | generic_reasoning | low | Plausible tendencies |
| `segment_month_effects` | Seasonal shift in the hidden segment mix (creates real train/test drift) | generic_reasoning | medium | Military PCS moves peak in summer; students move at term ends (June) and starts (Sept); families move in school breaks |
| `unit_base_logit` | Base log-odds that a booking buys from each product family (before profile/segment effects) | generic_reasoning | low | Calibrated once, before any model existed, so a local residential cart has a median of ~5 add-on lines (specialty and storage left rare) |
| `profile_effects` | Additive log-odds effects of observable profile features on each product family | generic_reasoning | low | Plausible business logic (stairs need stair protection, long moves need wrap and storage, etc.) |
| `segment_effects` | Additive log-odds effects of the HIDDEN segment on each product family | generic_reasoning | low | Plausible segment tastes; the model never sees the segment |
| `cooccurrence_lifts` | If family A is in the cart, raise the log-odds of family B by lift (applied in order, as a conditional second draw) | generic_reasoning | low | Items that go together in practice |
| `exclusive_units` | Families that can't both be in a cart (the first one wins) | company_kb | high | Day-of packing only exists when the customer did not book full packing (KB section 10) |
| `booking_noise_sd` | Per-booking shared log-odds noise (some customers just buy more) | generic_reasoning | low | Unobserved spending propensity |
| `item_noise_sd` | Per-booking, per-family log-odds noise | generic_reasoning | low | Idiosyncratic choices |
| `random_add_prob` | Chance a booking gets one extra random family (creates rare/odd combinations) | generic_reasoning | low | Upsells, data-entry quirks, odd requests |
| `random_drop_prob` | Chance each chosen item is dropped (cancelled line, forgotten) | generic_reasoning | low | Line items removed before the move |
| `size_class` | Groups home sizes into small/medium/large for variant choice and quantities | generic_reasoning | high | Convenience grouping |
| `variant_weights` | Which size/type variant a booking picks within a family, by size class | generic_reasoning | low | Bigger homes pick bigger units and wider wrap |
| `extra_variant_prob` | Chance a booking buys a second variant of the same family (storage is never 2 units) | generic_reasoning | low | Customers sometimes mix sizes |
| `quantity_rules` | Quantity per chosen item: median (by size class) x lognormal(sd), rounded to step, at least min. Labor items are hours. | generic_reasoning | low | Scaled to home size; box counts follow common 'boxes per room' rules of thumb |
| `storage_months` | Months of storage (or U-Box) rental; short-term vs long-term split by hidden segment | generic_reasoning | low | Students store over summer, seniors store long-term while downsizing |
| `hourly_rates` | Local hourly rate with truck, cash/check | company_kb | high | Company KB, section 6 (local move rates) |
| `payment_surcharge_per_hour` | Hourly rate change by payment method | company_kb | high | Company KB, section 6 |
| `min_labor_hours` | Minimum labor hours on every job (travel separate) | company_kb | high | Company KB, section 7 |
| `travel_time_multiplier` | Travel billed at this multiple of drive time between pickup and drop-off | company_kb | high | Company KB, section 7 (double drive time under California rules) |
| `labor_hours_range` | Typical labor hours by size (uniform draw). Residential from the KB; offices are guesses (the KB gives no office ranges). | company_kb | high | Company KB, section 8 (typical labor time) |
| `labor_only_hours_factor` | Labor-only jobs (load or unload a customer truck) take this share of full-move hours | generic_reasoning | low | No truck driving or two-ended handling |
| `extra_hours` | Extra labor hours for stairs and heavy items, and lognormal noise on total hours | generic_reasoning | low | KB says extra time applies for stairs and heavy items; amounts are guesses |
| `specialty_fee` | Flat fee per group of up to 3 specialty or heavy items | company_kb | high | Company KB, section 3 |
| `storage_free_month_after` | Storage longer than this many months gets the first month free | company_kb | high | Company KB, section 2 |
| `long_distance_rate_per_lb` | Flat long-distance price = weight x rate (linear in distance), with a minimum charge (the KB gives no rate) | public | low | 2026 cost guides put long-distance moves at about $0.50-$0.80 per lb; minimum charge is a guess (https://www.eeward.com/blog/how-much-does-a-long-distance-move-cost ; https://www.caddymoving.com/blogs/caddy-shack-blog/moving-costs-in-2026-complete-breakdown-estimates-and-ways-to-save) |
| `box_prices` | Price per box | public | medium | 2026 retailer comparison (small ~$1.08-$1.78, large ~$2.26-$3.00) (https://blog.hireahelper.com/tag/price-comparison/ ; https://greatguysmoving.com/blog/everything-you-need-to-know-about-moving-boxes) |
| `wardrobe_box_prices` | Price per wardrobe box (sold to keep) | public_derived | medium | Public prices range about $8-$25 per wardrobe box; sizes interpolated (https://blog.hireahelper.com/tag/price-comparison/) |
| `plastic_wrap_prices` | Price per roll, scaled by width from a public 20 in roll price | public_derived | medium | U-Haul lists 20 in x 1000 ft stretch wrap at $29.25 (https://www.uhaul.com/MovingSupplies/Packing-Supplies/20-Movers-Stretch-Plastic-Wrap?id=740) |
| `other_supply_prices` | Price per unit (roll, bundle or pack) | generic_reasoning | low | No reliable public price found for speed packs; others are typical retail |
| `protection_equipment_prices` | Price per unit for protection, straps, and adding a store pickup to a move | generic_reasoning | low | Typical retail; store pickup about 45 minutes of crew time |
| `storage_monthly_rates` | Monthly rent. Storage from San Diego County climate-controlled averages (5x10, 10x15 interpolated); U-Box rate is a guess. | public_derived | medium | RentCafe San Diego County climate-controlled: 5x5 ~$86-96, 10x10 ~$206-213, 10x20 ~$357 (https://www.rentcafe.com/climate-controlled-storage-units/us/ca/san-diego-county/92154/) |

**Count by source type:** company_kb: 16, generic_reasoning: 55, public: 3, public_derived: 6
