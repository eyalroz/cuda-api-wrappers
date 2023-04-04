/**
 * @file
 *
 * @brief A python-inspired container wrapper which allows for iterating a structure while keeping note of the element index.
 *
 * @note This mechanism allows for code such as the following:
 *
 * @author 2013, Sebastian Jeltsch <sjeltsch@kip.uni-heidelberg.de>
 * @author 2021, Eyal Rozenberg <eyalroz1@gmx.com>
 *
 * @copyright
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *  
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *  
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 */

#include <iostream>
#include <vector>
#include <tuple>

// Note: Does not support construction from an rvalue-reference
template <typename Container>
class enumerator
{
	using container_value_type = typename Container::value_type;
	using size_type = typename Container::size_type;
public:
	// The return value of the operator* of the iterator, this
	// is what you will get inside of the for loop
	struct item
	{
		size_type index;
		container_value_type& item;
	};
	using value_type = item;

	struct const_item
	{
		size_type index;
		const container_value_type& item;
	};

	using const_value_type = const_item;


	// Custom iterator with minimal interface
	struct iterator {
		using difference_type = std::ptrdiff_t;
		using value_type = enumerator<Container>::value_type;
		using pointer = value_type*;
		using reference = value_type&;
		using iterator_category = std::forward_iterator_tag;

		iterator(typename Container::iterator _it, size_type counter=0) : it(_it), counter(counter) {}

		iterator operator++() { return iterator(++it, ++counter); }

		bool operator!=(iterator other) { return it != other.it; }

		typename Container::iterator::value_type item() { return *it; }

		value_type operator*() { return value_type{counter, *it}; }

		size_type index() { return counter; }

	protected:
		typename Container::iterator it;
		size_type counter;
	};

	// TODO: Reduce DRY here...
	struct const_iterator {
		using difference_type = std::ptrdiff_t;
		using value_type = enumerator<Container>::const_value_type;
		using pointer = const_value_type*;
		using reference = const_value_type&;
		using iterator_category = std::forward_iterator_tag;

		const_iterator(typename Container::const_iterator _it, size_type counter=0) : it(_it), counter(counter) {}

		const_iterator operator++() { return const_iterator(++it, ++counter); }

		bool operator!=(const_iterator other) { return it != other.it; }

		typename Container::const_iterator::value_type item() { return *it; }

		const_value_type operator*() { return const_value_type{counter, *it}; }

		size_type index() { return counter; }

	protected:
		typename Container::const_iterator it;
		size_type counter;
	};

	enumerator(Container& container) : container_(container) {}

	iterator begin() { return iterator(container_.begin()); }
	iterator end()   { return iterator(container_.end()); }
	const_iterator cbegin() const { return const_iterator(container_.cbegin()); }
	const_iterator cend()   const { return const_iterator(container_.cend()); }

private:
	Container& container_;
};

// A templated free function allows you to create the wrapper class
// conveniently 
template <typename Container>
enumerator<Container> enumerate(Container & container)
{
	return enumerator<Container>(container);
}

