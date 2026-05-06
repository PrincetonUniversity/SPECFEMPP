#pragma once

namespace specfem {

class message_tag {
private:
  unsigned int connections_per_process;
  unsigned int messages_per_connection;
  unsigned int number_of_messages;
  int max_messages;

  unsigned int my_message_start;

  int sender_rank;
  int receiver_rank;

  bool initialized = false;
  bool is_pair_set = false;

public:
  message_tag(int max_messages);

  message_tag(int max_messages, unsigned int nprocesses,
              unsigned int connections_per_process,
              unsigned int messages_per_connection);

  int operator()(int sender_rank, int receiver_rank, int message_index) const;

  int operator()(int message_index) const;

  void set_pair(int sender_rank, int receiver_rank);

  void clear_pair();

  void initialize(unsigned int nprocesses, unsigned int connections_per_process,
                  unsigned int messages_per_connection);

  void reset();
};

} // namespace specfem
