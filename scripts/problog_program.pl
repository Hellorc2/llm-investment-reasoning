% Positive founder attributes
0.8::likelihood_of_success :- founder_has_deep_industry_expertise.
0.85::likelihood_of_success :- founder_has_prior_successful_startup_experience.
0.75::likelihood_of_success :- founder_has_relevant_education.
0.7::likelihood_of_success :- founder_has_leadership_at_top_companies.
0.8::likelihood_of_success :- founder_has_strong_investor_network.
0.7::likelihood_of_success :- founder_has_consistent_career_progression.
0.75::likelihood_of_success :- founder_has_specialized_skills_matching_startup.
0.7::likelihood_of_success :- founder_has_media_visibility.
0.8::likelihood_of_success :- founder_has_complementary_cofounders.
0.75::likelihood_of_success :- founder_has_prior_fundraising_experience.

% Negative founder attributes
0.3::likelihood_of_success :- founder_lacks_industry_expertise.
0.4::likelihood_of_success :- founder_is_first_time_entrepreneur.
0.3::likelihood_of_success :- founder_has_short_job_tenures.
0.35::likelihood_of_success :- founder_lacks_relevant_education.
0.3::likelihood_of_success :- founder_has_no_investor_network.
0.4::likelihood_of_success :- founder_is_solo_without_cofounders.
0.2::likelihood_of_success :- founder_has_unclear_startup_idea.
0.3::likelihood_of_success :- founder_has_no_prior_leadership_experience.
0.3::likelihood_of_success :- founder_spreads_focus_across_multiple_ventures.
0.4::likelihood_of_success :- founder_lacks_media_visibility.

% Positive startup attributes
0.8::likelihood_of_success :- startup_in_founder_core_expertise.
0.85::likelihood_of_success :- startup_has_product_market_fit.
0.7::likelihood_of_success :- startup_in_high_growth_industry.
0.8::likelihood_of_success :- startup_has_early_customer_traction.
0.75::likelihood_of_success :- startup_has_strong_differentiation.

% Negative startup attributes
0.3::likelihood_of_success :- startup_in_crowded_market_without_differentiation.
0.2::likelihood_of_success :- startup_lacks_clear_business_model.
0.4::likelihood_of_success :- startup_in_declining_industry.
0.3::likelihood_of_success :- startup_has_no_customer_validation.
0.2::likelihood_of_success :- startup_requires_expertise_founder_lacks.

% Additional positive founder attributes
0.7::likelihood_of_success :- founder_has_military_background.
0.65::likelihood_of_success :- founder_has_global_experience.
0.7::likelihood_of_success :- founder_worked_at_top_tier_companies.
0.65::likelihood_of_success :- founder_has_published_research.

% Additional negative founder attributes
0.4::likelihood_of_success :- founder_has_history_of_failed_startups.
0.3::likelihood_of_success :- founder_has_no_online_presence.
0.4::likelihood_of_success :- founder_has_unrelated_education.
0.3::likelihood_of_success :- founder_has_career_gaps.

% Funding and visibility attributes
0.7::likelihood_of_success :- startup_has_seed_funding.
0.8::likelihood_of_success :- startup_has_reputable_investors.
0.65::likelihood_of_success :- startup_has_press_coverage.

% Negative funding and visibility attributes
0.3::likelihood_of_success :- startup_has_no_funding_history.
0.35::likelihood_of_success :- startup_has_no_investor_backing.
0.4::likelihood_of_success :- startup_has_no_presence.

% Age and experience combinations
0.4::likelihood_of_success :- founder_over_40_no_prior_startups.
0.7::likelihood_of_success :- founder_under_30_with_expertise.
0.75::likelihood_of_success :- founder_has_both_corporate_and_startup_experience.

% Limited experience combinations
0.5::likelihood_of_success :- founder_has_only_corporate_experience.
0.4::likelihood_of_success :- founder_has_only_academic_background.
0.8::likelihood_of_success :- founder_has_hybrid_technical_business_skills.
