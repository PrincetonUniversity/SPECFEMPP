#include "message_tag.hpp"
#include "specfem/logger.hpp"

namespace specfem {

message_tag::message_tag(int max_messages)
    : connections_per_process(0), messages_per_connection(0),
      max_messages(max_messages), sender_rank(-1), receiver_rank(-1),
      initialized(false), is_pair_set(false) {}

message_tag::message_tag(int max_messages, unsigned int nprocesses,
                         unsigned int connections_per_process,
                         unsigned int messages_per_connection)
    : connections_per_process(connections_per_process),
      messages_per_connection(messages_per_connection),
      max_messages(max_messages), initialized(true), is_pair_set(false),
      sender_rank(-1), receiver_rank(-1) {
  if (nprocesses * connections_per_process * messages_per_connection >
      static_cast<unsigned int>(max_messages)) {
    specfem::Logger::error("message_tag: Total messages (" +
                           std::to_string(nprocesses * connections_per_process *
                                          messages_per_connection) +
                           ") exceed max_messages (" +
                           std::to_string(max_messages) + ")");
  }
}

int message_tag::operator()(int sender_rank, int receiver_rank,
                            int message_index) const {

  if (!initialized) {
    specfem::Logger::error("message_tag: message_tag not initialized");
  }

  if (message_index >= messages_per_connection) {
    specfem::Logger::error(
        "message_tag: message_index exceeds messages_per_connection");
  }
  return static_cast<int>(
      (static_cast<unsigned long>(sender_rank) * connections_per_process *
           messages_per_connection +
       receiver_rank * messages_per_connection + message_index) %
      max_messages);
}

int message_tag::operator()(int message_index) const {
  if (!initialized) {
    specfem::Logger::error("message_tag: message_tag not initialized");
  }
  if (!is_pair_set) {
    specfem::Logger::error("message_tag: sender/receiver pair not set");
  }
  if (message_index >= messages_per_connection) {
    specfem::Logger::error(
        "message_tag: message_index exceeds messages_per_connection");
  }
  return static_cast<int>(this->my_message_start + message_index) %
         max_messages;
}

void message_tag::set_pair(int sender_rank, int receiver_rank) {
  this->sender_rank = sender_rank;
  this->receiver_rank = receiver_rank;
  this->my_message_start =
      (static_cast<unsigned long>(sender_rank) * connections_per_process *
           messages_per_connection +
       receiver_rank * messages_per_connection);
  this->is_pair_set = true;
}

void message_tag::clear_pair() {
  this->sender_rank = -1;
  this->receiver_rank = -1;
  this->my_message_start = 0;
  this->is_pair_set = false;
}

void message_tag::initialize(unsigned int nprocesses,
                             unsigned int connections_per_process,
                             unsigned int messages_per_connection) {
  this->connections_per_process = connections_per_process;
  this->messages_per_connection = messages_per_connection;
  this->initialized = true;
}

void message_tag::reset() {
  this->connections_per_process = 0;
  this->messages_per_connection = 0;
  this->initialized = false;
  this->is_pair_set = false;
  this->my_message_start = 0;
  this->sender_rank = -1;
  this->receiver_rank = -1;
}

} // namespace specfem
