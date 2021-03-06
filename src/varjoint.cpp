/* evomodel.cpp
 *
 * This file is part of EvoModel.
 *
 * Copyright 2014 David B. Knoester.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <ea/evolutionary_algorithm.h>
#include <ea/representations/realstring.h>
#include <ea/generational_models/steady_state.h>
#include <ea/selection/rank.h>
#include <ea/selection/random.h>
#include <ea/cmdline_interface.h>
#include <ea/datafiles/fitness.h>
using namespace ealib;

#include "varjoint.h"
#include "analysis.h"

/*! Evolutionary algorithm definition.  EAs are assembled by providing a series of
 components (representation, selection type, mutation operator, etc.) as template
 parameters.
 */
typedef evolutionary_algorithm
< individual<realstring, joint_ks>
, ancestors::uniform_real
, mutation::operators::indel<mutation::operators::per_site<mutation::site::uniform_real> >
, recombination::asexual
, generational_models::steady_state<selection::random<with_replacementS>, selection::rank>
> ea_type;


/*! Define the EA's command-line interface.  Ealib provides an integrated command-line
 and configuration file parser.  This class specializes that parser for this EA.
 */
template <typename EA>
class cli : public cmdline_interface<EA> {
public:
    //! Define the options that can be parsed.
    virtual void gather_options() {
        
        add_option<REPRESENTATION_SIZE>(this);
        add_option<REPRESENTATION_MAX_SIZE>(this);
        add_option<REPRESENTATION_MIN_SIZE>(this);
        add_option<MUTATION_INSERTION_P>(this);
        add_option<MUTATION_DELETION_P>(this);
        add_option<MUTATION_INDEL_MAX_SIZE>(this);
        add_option<MUTATION_INDEL_MIN_SIZE>(this);
        
        add_option<POPULATION_SIZE>(this);
        add_option<MUTATION_PER_SITE_P>(this);
        add_option<MUTATION_UNIFORM_REAL_MIN>(this);
        add_option<MUTATION_UNIFORM_REAL_MAX>(this);
        add_option<STEADY_STATE_LAMBDA>(this);
        add_option<RUN_UPDATES>(this);
        add_option<RUN_EPOCHS>(this);
        add_option<CHECKPOINT_PREFIX>(this);
        add_option<RNG_SEED>(this);
        add_option<RECORDING_PERIOD>(this);
        
        add_option<EVOMODEL_M>(this);
        add_option<EVOMODEL_N>(this);
        add_option<EVOMODEL_MIN>(this);
        add_option<EVOMODEL_MAX>(this);
        add_option<EVOMODEL_MEAN1>(this);
        add_option<EVOMODEL_STD1>(this);
        add_option<EVOMODEL_MEAN2>(this);
        add_option<EVOMODEL_STD2>(this);
    }
    
    //! Define events (e.g., datafiles) here.
    virtual void gather_events(EA& ea) {
        add_event<datafiles::fitness_dat>(ea);
    };
};

// This macro connects the cli defined above to the main() function provided by ealib.
LIBEA_CMDLINE_INSTANCE(ea_type, cli);
